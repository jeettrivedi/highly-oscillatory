Asymptotic_Method_no_stat:=proc(f_t,g_t,a,b,p,omega_range::list)

	description "Computes Highly Oscillatory integrals with an rth order stationary point in the interval";
	local Q_A, f, g, h, k, mu, omega, rho, vals, xi;

	# Transforming the integral
	f:=x->f_t(x*(b-a)+a);
	g:=x->g_t(x*(b-a)+a);

	# Calculating the coefficients
	rho[0]:= f(x);
	for k from 1 to p-1 do
		rho[k]:= diff(rho[k-1]/diff(g(x),x),x);
	od;

	# Calculating the integral values
	return [ seq(evalf(eval(-(b-a)*add(1/(-I*w)^(m+1)*(eval(exp(I*w*g(x))*rho[m]/diff(g(x),x),x=1)-eval(exp(I*w*g(x))*rho[m]/diff(g(x),x),x=0)),m=0..p-1),w=omega)),omega = omega_range) ]:
	
end proc;
