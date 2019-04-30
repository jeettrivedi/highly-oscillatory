Filon_Method_Stat_Pts:=proc(f_t::algebraic,g_t::algebraic,a,b,N::integer,s_t,xi_t,r::integer,omega_range::list)
	description "Computes highly oscillatory integrals of the form 
	Int(f(x)*exp(I*w*g(x)),x=a..b).
	using Filon-Hermite integration. The method requires that g(x) have only 1 stationary point in [a,b]";

	local appended_zero,Int_f, array_of_funcs, f, f_tau, g, i, interpolant_coeffs, j, p, s, tau, vals, wts, wts_pre_calc, xi, xy;

	Digits:=16:
	# Transforming the integral
	f:=x->(b-a)*f_t(x*(b-a)+a);
	g:=x->g_t(x*(b-a)+a);
	xi:=(xi_t-a)/(b-a);

	# Generating interpolation points 
	tau := Array([seq((1+cos(Pi*i/(N-1)))*(1/2), i = N-1 .. 0, -1)]);

	# Adding stat point as a node if it is not a node already
	appended_zero := false:
	if(not(has(evalf(tau),evalf(xi)))) then 
		tau := ArrayTools[Append](tau,xi):
		appended_zero := true:
	end if:

	# make confluency vector
	s:=Array([seq(1,j=1..ArrayTools[Size](tau)[2])]):
	s[1] := s_t:

	if(appended_zero) then
		s[ArrayTools[Size](tau)[2]] := s_t*(r+1):
		s[ArrayTools[Size](tau)[2]-1] := s_t:
	else
		s[ArrayTools[Size](tau)[2]] := s_t:
		s[(ArrayTools[Size](tau)[2]+1)/2] := s_t*(r+1):
	end if;

	# Hermite interpolation
	xy:=evalf([seq([tau[m], seq(eval(diff(f(x), [x$j]), x = tau[m]), j = 0 .. s[m]-1)], m = 1 .. ArrayTools[Size](tau)[2])]);
	p:=x->add(alpha[i]*x^(i-1), i = 1 .. add(s));
	interpolant_coeffs := solve({seq(seq(eval(diff(p(x), [x$i]), x = xy[j][1]) = xy[j][i+2],i = 0 .. s[j]-1 ), j = 1 .. ArrayTools[Size](tau)[2])}):
	p:=x->add(rhs(interpolant_coeffs[i])*x^(i-1), i=1..add(s));

	# Calculating the moments
	wts:=map(c->value(c),Array([seq(Int(x^i*exp(I*omega*g(x)), x = 0 .. 1), i = 0 .. add(s)-1)])):

	Int_f := p(0)*wts[1]+add(rhs(interpolant_coeffs[i])*wts[i], i = 2 .. add(s)):

	# Calculation of the integral
	return [seq(evalf(subs(omega = w, Int_f)),w=omega_range)]:

end proc:
