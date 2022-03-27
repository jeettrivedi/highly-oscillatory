read "BHIP.mpl":

unroll := proc(p,s)
	local ret_arr,i,j,k:
	ret_arr := [ ]:
	for i from 1 to nops(p) do
		for j from 1 to s[i] do		
			ret_arr := [op(ret_arr),p[i,j]]:
		od:	
	od:
	return ret_arr:
end proc:

Levin_Hermite := proc(F,G,tau_in,N_in,s_in,omega_range)
	local D, Levin, P, Sum_s, a, b, d, gam_t, i, j, k, l, p, p_rhs, p_t, s, s_i, tau, w, Coeff_matrix, num_nodes, tau_unrolled:
	Levin := Matrix(nops(omega_range),1,fill=0):


	num_nodes := N_in:
	a := tau_in[1]:
	b := tau_in[2]:
	tau := [seq((a+b)/2+(b-a)*cos(Pi*(num_nodes-k)/(num_nodes-1))/2, k = 1 .. num_nodes)]:
	s := [seq(1,k=1..num_nodes)]:
	s[1] := s_in:
	s[num_nodes] := s_in:
	d := add(s[k], k=1 .. nops(s)):
	

	p := []:
	p_rhs := []:

	for i from 1 to nops(tau) do 
			p := [op(p),[seq(eval(diff(F(x),[x$(j-1)]),x=tau[i])/factorial(j),j=1..s[i])]]:
			p_rhs := [op(p_rhs),[seq(eval(diff(F(x),[x$(j-1)]),x=tau[i]),j=1..s[i])]]:
	od:	


	( p_t, gam_t, D ) := BHIP( p, tau, t, 'Dmat'=true ):
	p := unroll(p,s):
	p_rhs := unroll(p_rhs,s):
	tau_unrolled := Matrix(d,1,fill=0):
	k := 1:
	for i from 1 to num_nodes do 
		for j from 1 to s[i] do 
			tau_unrolled[k] := tau[i]:
			k := k + 1:
		od:
	od:
	Sum_s := Statistics[CumulativeSum]([0,op(s)])[1..nops(s)]:	

	i := 1:
	for w in omega_range do
		Coeff_matrix := Matrix(d,d,fill=0):
		k := 1:
		for j in Sum_s do
			j := convert(j,rational):
			for s_i from 0 to s[k]-1 do
				Coeff_matrix[j+1+s_i] := D[j+1+s_i]*factorial(s_i):
				for l from 1 to s_i do
	 				Coeff_matrix[j+1+s_i] := Coeff_matrix[j+1+s_i]+I*w*D[j+l]*eval(diff(G(x),[x$(s_i-l+1)]),x=tau[k])*factorial(l-1)*combinat[numbcomb](s_i,l):
				od:
				Coeff_matrix[j+s_i+1] := Coeff_matrix[j+s_i+1] + Matrix(d,d,shape=identity)[j+1]*eval(diff(G(x),[x$(s_i+1)]),x= tau[k])*I*w:
			od:
			k:= k + 1:
		od:
	
	

	P := LinearAlgebra[LinearSolve](evalf(Coeff_matrix),LinearAlgebra[Transpose](convert(p_rhs,Matrix))):
	Levin[i] := (P[LinearAlgebra[Dimensions](P)[1]-s[nops(s)]+1]*exp(I*w*G(b))-P[1]*exp(I*w*G(a))):
	i := i + 1:
	od:

	return convert(Levin,list):
end proc:

(*

N := 4:
tau := [seq(cos(Pi*(N-k)/(N-1)), k = 1 .. N)]:
s := [seq(2,k=1..N)]:
f := x -> cos(x):
g := x -> (x):
w_range := [seq(k,k=2 .. 2)]:
# F,G,tau_in,N_in,s_in,omega_range
intg := Levin_Hermite(f,g,[-1,1],5,2,w_range):
exact_vals := [seq(evalf(int(f(x)*exp(I*w*g(x)),x=tau[1]..tau[N])),w=w_range)]:
err := [seq(abs(intg[k]-exact_vals[k]),k=1..nops(exact_vals))]:
print(exact_vals):
print(evalf(intg)):
print(evalf(err)):
*)
