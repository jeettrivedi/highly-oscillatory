Digits := 16:

B := (x,i,n,a,b) -> combinat[numbcomb](n,i)*(x-a)^i*(b-x)^(n-i)/(b-a)^n:

Coeff_to_poly := proc(coeff_list::list,x,a,b)
	return add(coeff_list[j+1]*B(x,j,nops(coeff_list)-1,a,b),j=0..nops(coeff_list)-1):
end proc:

Bern_Interpolate := proc(tau::list,ftau::list,a,b)
	local n,A:
	n := nops(tau)-1:
	A := Matrix([seq([seq(B(tau[i],j-1,n,a,b),i=1..n+1)],j=1..n+1)]):
	return convert(LinearAlgebra[LinearSolve](A,Vector(ftau)),list):
end proc:

Bernstein_diff_matrix := proc(n,a,b)
	local D,i:
	D := Matrix(n+1,n+1,fill=0):
	D[1,1] := -n:
	D[1,2] := n:
	D[n+1,n+1] := n:
	D[n+1,n] := -n:
	for i from 2 to n do
		D[i,i] := 2*(i-1)-n:
		D[i,i-1] := -i+1:
		D[i,i+1] := n-i+1:
	od:
	return D/(b-a):
end proc:

Mul_Operator_Matrix := proc(f::list)
	local n,w,i,j:
	n := nops(f)-1:
	w := Matrix(2*n+1,n+1,fill=0):
	for i from 0 to 2*n do
		for j from max(0,i-n) to min(n,i) do
			w[i+1,j+1] :=  combinat[numbcomb](n,j)*combinat[numbcomb](n,i-j)/combinat[numbcomb](2*n,i)*f[i-j+1]:
		od:
	od:
	return w:
end proc: 

Levin_Bern_int := proc(F::algebraic,G::algebraic,tau::list,omega_range::list) 
	
	local a,b,n,coeff_f,coeff_one,coeff_g:
	n := nops(tau)-1:
	a := tau[1]:
	b := tau[n+1]:

	coeff_f := Bern_Interpolate([seq((1+cos(Pi*i/(2*n)))*(1/2), i = (2*n+1)-1 .. 0, -1)],map(F,[seq((1+cos(Pi*i/(2*n)))*(1/2), i = 2*n .. 0, -1)]),a,b):
	coeff_one := [seq(0,i=0..n)]:
	coeff_g := Bern_Interpolate(tau,map(G,tau),a,b):
	D := Bernstein_diff_matrix(n,a,b):
	#coeff_dg := coeff_g:
	print(D):
	
end proc: 

# print((Mul_Operator_Matrix([1,2,3,4,5]))):
#Bern_Interpolate([0,1/4,1/2,3/4,1],[1,2,3,4,5],0,1);
#print(Coeff_to_poly([1,2,3,4,5],x,3,4));
# print(Bernstein_diff_matrix(5,0,1)):

f := x -> cos(x):
g := x -> (x-3)**3:
Levin_Bern_int(f,g,[-1,0.1],[seq(i,i=1..100)]):