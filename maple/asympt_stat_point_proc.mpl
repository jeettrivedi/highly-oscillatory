Asymptotic_Method_stat_points:=proc(f_t,g_t,a,b,xi_t,r,p,omega_range::list)

	# description "Computes Highly Oscillatory integrals of the form
	# int(f(x)*exp(I*w*g(x)),x=a..b)
	# with an rth order stationary point in the interval and w in omega_range":

	local f,g,xi,mu,rho,k,h,vals,omega:

	Digits := 16:

	# Transforming the integral
	f:=x->f_t(x*(b-a)+a);
	g:=x->g_t(x*(b-a)+a);
	xi:=(xi_t-a)/(b-a);

	# The ``simpler'' highly oscillatory integrals 
	mu := (n, omega, xi)->int((x-xi)^n*exp(I*omega*g(x)), x = 0..1);

	# Calculating the coefficients
	rho[0]:= F(x);
	for k from 1 to p-1 do
		rho[k]:= Diff((rho[k-1]-add((Limit((Diff(rho[k-1], [x$j]))/factorial(j), x = xi))*(x-xi)^j, j = 0 .. r-1))/(Diff(G(x), x)), x);
	od:

	for k from 0 to p-1 do
		rho[k] := simplify(value(eval(rho[k], {F(x) = f(x), G(x) = g(x)})));
	od:

	return [ seq( evalf(value(add(evalf(mu(j, omega, xi))*add((Limit(value(diff(rho[m], [x$j])), x = xi))/(-I*omega)^m, m = 0 .. p-1)/factorial(j), j = 0 .. r-1))-add((exp(I*omega*g(1))*(map(w->eval(w, x = 1),rho)[m]-map(w->limit(w, x = xi), rho)[m])/(eval(diff(g(x), x), x = 1))-exp(I*omega*g(0))*(map(w->eval(w, x = 0),rho)[m]-map(w->limit(w, x = xi),rho)[m])/(eval(diff(g(x),x), x=0)))/(-I*omega)^(m+1), m = 0 .. p-1))*(b-a), omega=omega_range) ]:


end proc;
