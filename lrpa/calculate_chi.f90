
subroutine calculate_chi_ph(chi, e, u, beta, na, nk, norb, nb)
    implicit none

    ! Arguments
    integer, intent(in) :: na, nk, norb, nb
    real(8), intent(in) :: e(na, nk, nb), beta
    complex(8), intent(in) :: u(na, nk, norb, nb)
    complex(8), intent(inout) :: chi(2,2,2,2)


    ! Local variables
    integer :: k, a, b, n, m, i, j
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
                                chi(a,b,b,a) = chi(a,b,b,a) + factor * &
                                    conjg(u(a, k,i, n)) * u(a, k,j, n) * &
                                    conjg(u(b, k,j, m)) * u(b, k,i, m)
                                            
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
end subroutine calculate_chi_ph
