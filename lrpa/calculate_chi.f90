
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
