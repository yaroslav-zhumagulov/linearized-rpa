
subroutine calculate_chi_ph(chi, e, u, beta, na, nk, norb, nb)
    implicit none

    ! Arguments
    integer, intent(in) :: na, nk, norb, nb
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(norb, na, nk, nb)   ! norb first: stride-1 for dot_product
    complex(8), intent(inout) :: chi(2,2,2,2)

    ! Local variables
    integer :: k, a, b, n, m
    real(8) :: factor, f_an, f_bm, delta_e
    real(8) :: f(na, nk, nb)
    complex(8) :: C

    ! Precompute Fermi-Dirac factors
    f = 1.0d0 / (1.0d0 + exp(beta * e))

    !$omp parallel do reduction(+:chi) private(a,b,n,m,C,factor,f_an,f_bm,delta_e) schedule(static)
    do k = 1, nk
        do a = 1, na
            do b = 1, na
                do n = 1, nb
                    f_an = f(a, k, n)
                    do m = 1, nb
                        f_bm = f(b, k, m)
                        delta_e = e(b, k, m) - e(a, k, n)

                        if (abs(delta_e) < 1.0d-10) then
                            factor = beta * f_an * (1.0d0 - f_an)
                        else
                            factor = (f_an - f_bm) / delta_e
                        end if

                        ! C = sum_i conj(u(i,a,k,n)) * u(i,b,k,m)
                        ! dot_product conjugates first arg for complex arrays
                        C = dot_product(u(:, a, k, n), u(:, b, k, m))
                        chi(a,b,b,a) = chi(a,b,b,a) + factor * (real(C)**2 + aimag(C)**2)

                    end do
                end do
            end do
        end do
    end do
    !$omp end parallel do

end subroutine calculate_chi_ph


subroutine calculate_chi_pp(chi, e, u, e_inv, u_inv, beta, na, nk, norb, nb)
    implicit none

    ! Arguments
    integer, intent(in) :: na, nk, norb, nb
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(na, nk, norb, nb)

    real(8), intent(in) :: e_inv(na, nk, nb)
    complex(8), intent(in) :: u_inv(na, nk, norb, nb)

    complex(8), intent(inout) :: chi(2,2,2,2)

    ! Local variables
    integer :: k, a, b, n, m, i, j
    integer:: sa,sb,sc,sd
    real(8) :: factor, fa, fb

    ! Loop over dimensions
    do k = 1, nk
        do a = 1, na
            do b = 1, na
                do n = 1, nb
                    do m = 1, nb

                        f = 1.0d0 / (1.0d0 + exp(beta * e(a, k, n)))
                        f_inv = 1.0d0 / (1.0d0 + exp(beta * e_inv(b, k, m)))
                        factor = (1.0d0 - f - f_inv) / (-e_inv(b, k, m) - e(a, k, n))

                        ! Accumulate into chi
                        do i = 1, norb
                            do j = 1, norb
                                chi(a,b,b,a) = chi(a,b,b,a) + factor * &
                                    conjg(u(a, k, i, n)) * u(a, k, j, n) * &
                                    conjg(u_inv(b, k,i, m)) * u_inv(b, k, j, m)
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine calculate_chi_pp


! Static chi^{tau,tau'}(q; G, G') with valley treated as pseudospin.
!
! Arguments
!   chi         (na, na, ng, ng) complex -- accumulates the sum; normalise by nk in Python
!   e           (na, nk, nb)     real    -- band energies relative to mu
!   u           (norb, na, nk, nb) complex -- eigenvectors, orbital index first (stride-1)
!   beta                         real    -- inverse temperature
!   kq_idx      (nk)             integer -- 1-based index of k+q on the k-grid
!   G_shift_idx (ng, norb)       integer -- 1-based shifted orbital index; 0 = out of basis
!   na, nk, norb, nb, ng         integer -- dimensions
!
subroutine calculate_chi_q(chi, e, u, beta, kq_idx, G_shift_idx, na, nk, norb, nb, ng)
    implicit none

    integer, intent(in) :: na, nk, norb, nb, ng
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(norb, na, nk, nb)
    integer, intent(in) :: kq_idx(nk), G_shift_idx(ng, norb)
    complex(8), intent(inout) :: chi(na, na, ng, ng)

    real(8)    :: f(na, nk, nb), f_an, f_bm, delta_e, w_nm
    complex(8), allocatable :: Fmat(:,:,:), ubconj(:,:), uaT(:,:), chi_block(:,:), Fvec(:)
    integer    :: k, kq, a, b, ig, jg, alpha, alpha_G, m, n

    f = 1.0d0 / (1.0d0 + exp(beta * e))

    !$omp parallel reduction(+:chi) &
    !$omp& private(k, kq, a, b, ig, jg, alpha, alpha_G, m, n, &
    !$omp&         f_an, f_bm, delta_e, w_nm, Fmat, ubconj, uaT, chi_block, Fvec)
        allocate(Fmat(nb, nb, ng), ubconj(nb, norb), uaT(nb, norb), chi_block(ng, ng), Fvec(ng))

        !$omp do schedule(static)
        do k = 1, nk
            kq = kq_idx(k)
            do a = 1, na
                do b = 1, na

                    ! Cache-friendly local copies
                    ! ubconj(m, alpha) = conj(u(alpha, b, kq, m))   -- column stride-1 in m
                    ! uaT(n, alpha)    = u(alpha, a, k, n)           -- column stride-1 in n
                    do m = 1, nb
                        ubconj(m, :) = conjg(u(:, b, kq, m))
                    end do
                    do n = 1, nb
                        uaT(n, :) = u(:, a, k, n)
                    end do

                    ! Form factors
                    ! Fmat(m, n, ig) = sum_{alpha: G_shift_idx(ig,alpha)>0}
                    !                  ubconj(m, alpha) * uaT(n, alpha_G)
                    Fmat(:, :, :) = (0.0d0, 0.0d0)
                    do ig = 1, ng
                        do alpha = 1, norb
                            alpha_G = G_shift_idx(ig, alpha)
                            if (alpha_G == 0) cycle
                            do n = 1, nb
                                ! Vector update: Fmat(:,n,ig) += ubconj(:,alpha) * uaT(n,alpha_G)
                                Fmat(:, n, ig) = Fmat(:, n, ig) + ubconj(:, alpha) * uaT(n, alpha_G)
                            end do
                        end do
                    end do

                    ! Accumulate chi_block(ig, jg) = sum_{n,m} Fmat(m,n,ig) * w(n,m) * conj(Fmat(m,n,jg))
                    chi_block(:, :) = (0.0d0, 0.0d0)
                    do n = 1, nb
                        f_an = f(a, k, n)
                        do m = 1, nb
                            f_bm = f(b, kq, m)
                            delta_e = e(b, kq, m) - e(a, k, n)
                            if (abs(delta_e) < 1.0d-10) then
                                w_nm = beta * f_an * (1.0d0 - f_an)
                            else
                                w_nm = (f_an - f_bm) / delta_e
                            end if
                            ! Rank-1 Hermitian update: chi_block += w * Fvec * Fvec^H
                            Fvec(:) = Fmat(m, n, :)
                            do jg = 1, ng
                                chi_block(:, jg) = chi_block(:, jg) + &
                                    w_nm * conjg(Fvec(jg)) * Fvec(:)
                            end do
                        end do
                    end do
                    chi(a, b, :, :) = chi(a, b, :, :) + chi_block

                end do
            end do
        end do
        !$omp end do
        deallocate(Fmat, ubconj, uaT, chi_block, Fvec)
    !$omp end parallel

end subroutine calculate_chi_q
