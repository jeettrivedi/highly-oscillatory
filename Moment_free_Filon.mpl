Moment_free_filon := proc(F,G,tau_in::list,xi,r,s_in,N_in,w_range)
	local L_phi, phi, N, Q_F, a, b, f, f_ap, g, i, n, varphi, s, s_t, tau, vals, xi_t, xy, func_arr_l, func_arr_r, int_coeffs, s_end_pts, s_stat_pt, stat_pt_pos, interpolant_coeffs_free_var:

    Digits := 16:
    _Envsignum0 := 0:

    # Transforming the integral to [-1,1]
    a := tau_in[1]:
    b := tau_in[2]:
	f := x -> F(x*(b-a)/2+(b+a)/2):
    g := x -> G(x*(b-a)/2+(b+a)/2):
    xi_t := (2*xi-a-b)/(b-a):

    # setting the confluencies appropriately
    N := N_in:
    s_t := 1:
    s_end_pts := s_in:
    s_stat_pt := (2*s_end_pts-1)*(r-1):
    tau := Array([seq((cos(Pi*i/(N-1))), i = N-1 .. 0, -1)]);
    
    # Adding stat point as a node if it is not a node already
    if(not(has(evalf(tau),xi_t))) then 
        tau := sort((ArrayTools[Append](tau,xi_t))):
    end if:
        
    stat_pt_pos := 2:
    for i from 2 to nops(tau)-1 do 
        if(evalf(tau[i])=xi_t) then 
            break:
        else 
            stat_pt_pos := stat_pt_pos + 1:
        end if
    od:
    
    # make confluency vector
    N := ArrayTools[Size](tau)[2]:
    s := Array([seq(s_t,j=1..N)]):
    s[stat_pt_pos] := s_stat_pt:
    s[1] := s_end_pts:
    s[N] := s_end_pts:
    n := add(s[i],i=1..N);
    
    # Basis functions and interpolation
    phi := (x_t,r_t,k_t,g_t)->eval(piecewise(x<0,piecewise(modp(r_t,2)=0,(-1)^k_t,modp(r_t,2)=1,(-1)^k_t*exp(-(1+k_t)/r_t*I*Pi)),x>0,-1)*w^(-(k_t+1)/r_t)/r_t*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);

    L_phi := (x_t,r_t,k_t,g_t) -> signum(x_t)^(r_t+k_t+1)*abs(g_t)^((k_t+1)/r_t-1)*diff(g_t,x)/r_t:
   
    f_ap := x -> add(c[j]*L_phi(x,r,j-1,g(x)),j=1..n):

    xy:=evalf([seq([tau[m], seq(eval(diff(f(x), [x$j]), x = tau[m]), j = 0 .. s[m]-1)], m = 1 .. N)]):

    interpolant_coeffs_free_var := fsolve([seq(seq(limit(diff(f_ap(x), [x$i]), x = xy[j][1],right) = xy[j][i+2],i = 0 .. s[j]-1 ), j = 1 .. N)]);

    int_coeffs := [seq(rhs(interpolant_coeffs_free_var[j]),j=1..n)];

    # Evaluation of the integral 
    func_arr_r := [ seq(phi(1,r,k,g(x))*exp(I*w*g(1)),k=0..n-1) ]:
    func_arr_l := [ seq(phi(-1,r,k,g(x))*exp(I*w*g(-1)),k=0..n-1) ]:

    Q_F := add(int_coeffs[j+1]*(func_arr_r[j+1]-func_arr_l[j+1]),j=0..n-1):

    vals := [(b-a)/2*seq(evalf(eval(Q_F,w=w_range[j])),j=1..nops(w_range))]:

    return vals:

end proc:
