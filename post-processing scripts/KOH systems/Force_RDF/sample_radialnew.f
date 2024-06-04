      Subroutine Sample_Radialnew(Ichoise,Temp)
      Implicit None

      Include 'system.inc'

Cccccccccccccccccccccccccccccccccccccccccccccccccc
C     Samples The Radial Distribution Function   C
Cccccccccccccccccccccccccccccccccccccccccccccccccc

      Integer I,J,Maxx,Ichoise,K

      Parameter(Maxx = 5000)

      Double Precision Ggt,Gg(Maxx),Delta,R,P,Ff,Dx,Dy,Dz,Temp

      Parameter (Delta  = 0.01d0)
      
      Save Ggt,Gg

      If(Ichoise.Eq.1) Then
         Do I=1,Maxx
            Gg(I) = 0.0d0                                                                                                              C f64
         Enddo

         Ggt = 0.0d0
      
      Elseif(Ichoise.Eq.2) Then

         Ggt = Ggt + 1.0d0

C     Loop Over All Pairs

         Do I=1,NumberOfParticles-1
            Do J=I+1,NumberOfParticles

               Dx = Rxx(I) - Rxx(J)
               Dy = Ryy(I) - Ryy(J)
               Dz = Rzz(I) - Rzz(J)
 
               If (Dx.Gt.HalfBox) Then
                  Dx = Dx - Box
               Elseif (Dx.Lt. - HalfBox) Then
                  Dx = Dx + Box
               Endif
 
               If (Dy.Gt.HalfBox) Then
                  Dy = Dy - Box
               Elseif (Dy.Lt. - HalfBox) Then
                  Dy = Dy + Box
               Endif
 
               If (Dz.Gt.HalfBox) Then
                  Dz = Dz - Box
               Elseif (Dz.Lt. - HalfBox) Then
                  Dz = Dz + Box
               Endif

               R = Dsqrt(Dx*Dx + Dy*Dy + Dz*Dz)

               If(R.Lt.HalfBox) Then
                  Ff = (Fxx(I)*Dx + Fyy(I)*Dy + Fzz(I)*Dz)/(R*R*R)

                  Do K=1,Maxx
                     P = (Dble(K)-0.5d0)*Delta

                     If(P.Lt.Halfbox.And.P.Gt.R) Gg(K) = Gg(K) + Ff
                  Enddo
               Endif
            Enddo
         Enddo

      Else

C     Write Results To Disk

         Open(21,File="Radialnew",Status="Unknown")
         Do I=1,Maxx-1

            P = (Box**3)/
     &           (Ggt*Temp*8.0d0*Datan(1.0d0)*
     &           (Dble(NumberOfParticles)**2))
            
            If((Dble(I)-0.5d0)*Delta.Lt.Halfbox)
     &           Write(21,*) ((Dble(I)-0.5d0)*Delta),Gg(I)*P
         Enddo
         Close(21)
      Endif

      Return
      End
