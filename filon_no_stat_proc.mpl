Filon_Method_No_Stat:=proc(f_t::algebraic,g_t::algebraic,a::numeric,b::numeric,N::integer,omega_range::list)
	description "Computes highly oscillatory integrals of the form 
	Int(f(x)*exp(I*w*g(x)),x=a..b).
	using Filon-Lagrange integration. The method requires that g(x) have no stationary points in the interval [a,b].";


	local Int_f, array_of_funcs, f, f_tau, g, i, j, p, tau, vals, wts, wts_pre_calc;
	Digits:=16:

	# Transforming the integral
	f := x->f_t(x*(b-a)+a);
	g := x->g_t(x*(b-a)+a);

	# Generating interpolation points 
	tau := [seq((1+cos(Pi*i/(N-1)))*(1/2), i = N-1 .. 0, -1)];
	f_tau := [seq(f(tau[i+1]), i = 0 .. N-1)];

	# Calculating the moments
	wts := [seq(int(x^i*exp(I*omega*g(x)), x = 0 .. 1), i = 0 .. N-1)];

	# Interpolation
	p := z->CurveFitting[PolynomialInterpolation](zip(`[]`, tau, f_tau), z);
	p := collect(p(z), z); 
	Int_f := (b-a)*((subs(z = 0, p)*wts[1]+add(coeff(p, z^(i-1))*wts[i], i = 2 .. N)));

	# Calculation of the integral
	return [ seq(evalf(subs(omega = w, Int_f)),w=omega_range) ]:

end proc:
