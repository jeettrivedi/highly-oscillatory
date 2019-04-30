Calc_Coefficients := proc(f_t,g_t,r_t::numeric)
	# Computes the interpolation coefficients by solving a system of equations

	local eq_sys,sols,c,rhs_eqs,L_phi:

	phi := (x_t,r_t,k_t,g_t)->eval(piecewise(x<0,piecewise(modp(r_t,2)=0,(-1)^k_t,modp(r_t,2)=1,(-1)^k_t*exp(-(1+k_t)/r_t*I*Pi)),x>0,-1)*w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);
	
	L_phi := (x_t,r_t,k_t,g_t)-> signum(x_t)^(r_t+k_t+1)*abs(g_t)^((k_t+1)/r_t-1)*diff(g_t,x)/r_t;

	rhs_eqs := [seq((diff(f_t,[x$j])),j=0..r_t-2)];

	rhs_eqs := map(h->limit(h,x=0),rhs_eqs);
	eq_sys := [seq(diff( add( c[k+1]*L_phi(x,r_t,k,g_t),k=0..r_t-2 ) ,[x$j]),j=0..r_t-2)];
	eq_sys := map(h->limit(h,x=0),eq_sys);
	sols := solve({seq(eq_sys[j+1]=rhs_eqs[j+1],j=0..r_t-2)},{seq(c[j+1],j=0..r_t-2)});

	return [seq(rhs(sols[i]),i=1..nops(sols))]:

end proc:


Moment_free_asymp := proc(F,G,tau_in::list,xi,r,p,w_range)
	
	local L_phi,mu_f,L,a,b,f,g,xi_t,sig_coef,sigma,func_arr_l,func_arr_r,L_func_arr_l,L_func_arr_r,sigma_arr_l,sigma_arr_r,l_term_2,r_term_2,sum_terms_2,l_term_1,r_term_1,sum_terms_1,phi:

	Digits := 16:
	_Envsignum0 := 0:
	
	# Transforming the integral to the correct interval
    a := tau_in[1]:
    b := tau_in[2]:
	f := x -> F(x*(b-a)/2+(b+a)/2):
    g := x -> G(x*(b-a)/2+(b+a)/2):
    xi_t := (2*xi-a-b)/(b-a):

    # basis functions
	phi := (x_t,r_t,k_t,g_t)->eval(piecewise(x<0,piecewise(modp(r_t,2)=0,(-1)^k_t,modp(r_t,2)=1,(-1)^k_t*exp(-(1+k_t)/r_t*I*Pi)),x>0,-1)*w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);
	
	L_phi := (x_t,r_t,k_t,g_t)-> signum(x_t)^(r_t+k_t+1)*abs(g_t)^((k_t+1)/r_t-1)*diff(g_t,x)/r_t;

	mu_f := (c,r_t,x,g_t) -> add(c[k+1]*phi(x,r_t,k,g_t),k=0..r_t-2);
	
	L := (F,G) -> diff(F,x)+I*w*diff(G,x)*F;

	# Computing the series
	sigma[0] := f(x):
	
	for k from 1 to p do
	    sig_coef[k-1] := Calc_Coefficients(sigma[k-1],g(x),r):
	    sigma[k] := diff( (sigma[k-1] - add( sig_coef[k-1][j+1]*L_phi(x,r,j,g(x)),j=0..r-2 ))/diff(g(x),x) ,x);
	od:
	sig_coef[p] := Calc_Coefficients(sigma[p],g(x),r):

	func_arr_r := [ seq(phi(1,r,k,g(x))*exp(I*w*g(1)),k=0..r-2) ]:
	func_arr_l := [ seq(phi(-1,r,k,g(x))*exp(I*w*g(-1)),k=0..r-2) ]:

	L_func_arr_l := [ seq(eval(L_phi(-1,r,k,g(x)),x=-1),k=0..r-2) ]:
	L_func_arr_r := [ seq(eval(L_phi(1,r,k,g(x)),x=1),k=0..r-2) ]:

	sigma_arr_l := [ seq( eval(sigma[k],x=-1) ,k=0..p) ]:
	sigma_arr_r := [ seq( eval(sigma[k],x=1) ,k=0..p) ]:

	l_term_2 := [ seq((sigma_arr_l[k+1] - add(sig_coef[k][j+1]*L_func_arr_l[j+1],j=0..r-2))*exp(I*w*g(-1))/eval(diff(g(x),x),x=-1),k=0..p) ]:

	r_term_2 := [ seq((sigma_arr_r[k+1] - add(sig_coef[k][j+1]*L_func_arr_r[j+1],j=0..r-2))*exp(I*w*g(1))/eval(diff(g(x),x),x=1),k=0..p) ]:

	sum_terms_2 := r_term_2 - l_term_2:

	l_term_1 := [ seq(add(sig_coef[k][j+1]*func_arr_l[j+1],j=0..r-2),k=0..p) ]:
	r_term_1 := [ seq(add(sig_coef[k][j+1]*func_arr_r[j+1],j=0..r-2),k=0..p) ]:
	
	sum_terms_1 := r_term_1 - l_term_1:

	Q_A := add(1/(-I*w)^k*sum_terms_1[k+1],k=0..p)-add(1/(-I*w)^(k+1)*sum_terms_2[k+1],k=0..p):

	return [ (b-a)/(2)*seq(evalf(eval(Q_A,w=w_range[j])),j=1..nops(w_range)) ]:

	eval(vals)

end proc:
