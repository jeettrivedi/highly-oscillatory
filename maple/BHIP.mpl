#
# BHIP: Barycentric Hermite Interpolation Program
#
# (c) Robert M. Corless, December 2007, August 2012
#
# Compute the barycentric form of the unique Hermite interpolant
# of the polynomial given by values and derivative values of
# p(t) at the nodes tau.
#
# CALLING SEQUENCES
#
#   ( p, gam ) := BHIP( flist, tau, t );
#   ( p, gam ) := BHIP( ftayl, tau, t, 'Taylor' = true, 'Denominator' = q );
#   ( p, gam, DD ) := BHIP( ftayl, tau, t, <opts>, 'Dmat'=true )
#
# Processing: local Laurent series.
#             This approach is different to that of
# Reference: C. Schneider & W.Werner, "Hermite
#            Interpolation: The Barycentric Approach",
#            Computing 46, 1991, pp 35-51.
#
BHIP := 
proc( pin::list, tau::list, t::name,
			{Taylor::truefalse:=true}, 
			{Conditioning::truefalse:=false}, 
			{Dmat::truefalse:=false},
			{Denominator::{algebraic,list}:=1} )
     local brks, d, DD, denr, dens, dgam, 
           dr, g, gam, ghat,h, i, irow, j,
           k, mu, n, numr, nums,
           p, P, q, r, rs, rt, s, smax, sq;

     n := nops(tau);
     if nops(pin) <> n then
        error "Mismatched size of node list and data list"
     end if;

     if nops(convert(tau,set)) < n then
        error "Nodes must be distinct, with confluency explicitly specified."
     end if;

     p := map(t -> `if`(t::list,t,[t]),pin); # singletons ok
     s := map(nops,p); # confluency
     smax := max(op(s));
     if smax = 0 then
       error "At least one piece of data is necessary."
     end if;
     d := -1 + add( s[i], i=1..nops(s) );  # degree bound
     p := `if`( Taylor, p, [seq([seq(p[i][j]/(j-1)!,j=1..s[i])],i=1..n)] );

     gam := Array( 1..n, 0..smax-1 ); #default 0
     if Conditioning then
        dgam := Array( 1..n, 0..smax-1, 1..n ); #default 0
     end if;

     # The following works for n>=1
     for i to n do
         if s[i] > 0 then # ignore empty lists
            h[i] := mul( (t-tau[j])^s[j], j = 1..i-1 )*
                    mul( (t-tau[j])^s[j], j=i+1..n );
            r[i] := series( 1/h[i], t=tau[i], s[i] );
            for j to s[i] do
               gam[i,s[i]-j] := coeff( r[i], t-tau[i], j-1) ; #op( 2*j-1, r[i] );
            end do;
            if Conditioning then
               # We could compose a series for 1/(t-tau[k]) with
               # what we know, but using the kernel function "series"
               # is likely faster.
               for k to i-1 do
                  dr[i,k] := series( s[k]/h[i]/(t-tau[k]), t=tau[i], s[i] );
                  for j to s[i] do
                    dgam[i, s[i]-j, k] := coeff( dr[i,k], t-tau[i], j-1 );
                  end do;
               end do;
               # We could reuse earlier series, and do one O(n^2)
               # computation to get gam[i,-1], but it's simpler to
               # use series (and likely faster because series is in the kernel)
               dr[i,i] := series( 1/h[i], t=tau[i], s[i]+1 );
               # We implicitly divide this by t-tau[i], and take
               # coefficients one higher.
               for j to s[i] do
                  dgam[i,s[i]-j,i] := j*coeff( dr[i,i], t-tau[i], j );
               end do;
               for k from i+1 to n do
                  dr[i,k] := series( s[k]/h[i]/(t-tau[k]), t=tau[i], s[i] );
                  for j to s[i] do
                    dgam[i, s[i]-j, k] := coeff( dr[i,k], t-tau[i], j-1 );
                  end do;
               end do;
            end if;
         end if;
     end do;

     if not (Denominator::algebraic and Denominator=1) then
        # adjust gam by folding in q
        if Denominator::list then
           if nops(Denominator)<>n then
              error "Denominator list (q) has the wrong length."
           end if;
           q :=`if`( Taylor, q, [seq([seq(q[i][j]/(j-1)!,j=1..s[i])],i=1..n)] );
        else
           ghat := Array( 1..n );
           for i to n do
              sq := series(Denominator,t=tau[i],s[i]);
              ghat[i] :=[seq(coeff(sq,t-tau[i],j),j=0..s[i]-1)];
           end do;
           q := [seq(ghat[i],i=1..n)];
        end if;
        ghat := Array( 1..n, 0..smax-1 );
        for i to n do
           for j from 0 to s[i]-1 do
              ghat[i,j] := add( gam[i,j+k]*q[i][k+1], k=0..s[i]-j-1 );
           end do;
        end do;
        gam := ghat;
     end if;
       
     P := mul( (t-tau[i])^s[i],i=1..n)*
          add(add(gam[i,j]/(t-tau[i])^(1+j)*
                  add(p[i][1+k]*(t-tau[i])^k, k=0..j),
                  j=0..s[i]-1),
              i=1..n );
            
     # Translated from Matlab.  Nearly working.         
     if Dmat then
       # Compute differentiation matrix
       DD := Matrix( d+1, d+1 );
       brks := [seq(add(s[j],j=1..i-1),i=1..nops(s))]; #cumsum([0,s.']);
       irow := 0;
       for k to n do
         # trivial rows
         for j to s[k]-1 do
           irow := irow+1;
           # next available row
           DD[irow,brks[k]+j+1] := j;  # result is in Taylor form
         end;
         # Nontrivial row
         irow := irow+1;
         for i in [seq(j,j=1..k-1),seq(j,j=k+1..n)] do
           for j to s[i] do
             g := 0;
             for mu from j-1 to s[i]-1 do
                 g := g + gam[i,mu]*(tau[k]-tau[i])^(j-2-mu);
             end;
             DD[irow,brks[i]+j] := g/gam[k,s[k]-1];
           end;
         end;
         DD[irow,brks[k]+2..brks[k]+s[k]] := -gam[k,0..s[k]-2]/gam[k,s[k]-1];
         # Final entry 
         DD[irow,brks[k]+1] := -add( DD[irow,brks[j]+1], j=1..nops(brks) );
         DD[irow,1..-1] := DD[irow,1..-1]*s[k];  # want Taylor form of derivative
       end;
     end if;

     return P, gam, `if`(Conditioning,dgam,NULL), `if`(Dmat,DD,NULL) ;
end proc:
