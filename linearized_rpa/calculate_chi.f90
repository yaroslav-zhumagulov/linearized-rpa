
subroutine calculate_chi_ph(chi, e, u, beta, na, nk, norb, nb,ns)
    implicit none

    ! Arguments
    integer, intent(in) :: na, nk, ns, norb, nb
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(na, nk, ns, norb, nb)
    complex(8), intent(inout) :: chi(2,2,2,2,2,2,2,2)


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
                        if (e(b, k, m) == e(a, k, n)) then
                            fa = 1.0d0 / (1.0d0 + exp(+beta * e(a, k, n)))
                            fb = 1.0d0 / (1.0d0 + exp(-beta * e(a, k, n)))
                            factor = beta * fa * fb
                        else
                            fa = 1.0d0 / (1.0d0 + exp(beta * e(a, k, n)))
                            fb = 1.0d0 / (1.0d0 + exp(beta * e(b, k, m)))
                            factor = (fa - fb) / (e(b, k, m) - e(a, k, n))
                        end if

                        ! Accumulate into chi
                        do i = 1, norb
                            do j = 1, norb
                                do sa = 1, ns
                                    do sb = 1, ns
                                        do sc = 1, ns
                                            do sd = 1, ns
                                                chi(sa,a,sb,b,sc,b,sd,a) = chi(sa,a,sb,b,sc,b,sd,a) + factor * &
                                                    conjg(u(a, k,sa, i, n)) * u(a, k,sd, j, n) * &
                                                    conjg(u(b, k,sc, j, m)) * u(b, k,sb, i, m)
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine calculate_chi_ph


subroutine calculate_chi_pp(chi, e, u, e_inv, u_inv, beta, na, nk, norb, nb,ns)
    implicit none

    ! Arguments
    integer, intent(in) :: na, nk, ns, norb, nb
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(na, nk, ns, norb, nb)

    real(8), intent(in) :: e_inv(na, nk, nb)
    complex(8), intent(in) :: u_inv(na, nk, ns, norb, nb)

    complex(8), intent(inout) :: chi(2,2,2,2,2,2,2,2)

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

                        fa = 1.0d0 / (1.0d0 + exp(beta * e(a, k, n)))
                        fb = 1.0d0 / (1.0d0 + exp(beta * e_inv(b, k, m)))
                        factor = (1.0d0 - fa - fb) / (-e_inv(b, k, m) - e(a, k, n))

                        ! Accumulate into chi
                        do i = 1, norb
                            do j = 1, norb
                                do sa = 1, ns
                                    do sb = 1, ns
                                        do sc = 1, ns
                                            do sd = 1, ns
                                                chi(sa,a,sb,b,sc,b,sd,a) = chi(sa,a,sb,b,sc,b,sd,a) + factor * &
                                                    conjg(u(a, k,sa, i, n)) * u(a, k,sd, j, n) * &
                                                    conjg(u_inv(b, k,sc, i, m)) * u_inv(b, k,sb, j, m)
                                            end do
                                        end do
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine calculate_chi_pp
