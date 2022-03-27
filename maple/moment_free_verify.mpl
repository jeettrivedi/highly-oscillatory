restart:
Digits := 15:

# The expression for phi. The basis elements are given by L(phi). We want to check whether the "nicer" expression, L_phi, given for L(phi) holds. 

L := (F,G) -> diff(F,x)+I*w*diff(G,x)*F;

_Envsignum0 := 0:
phi := (x_t,r_t,k_t,g_t)->eval(piecewise(x_t<0,piecewise(modp(r_t,2)=0,(-1)^k_t,modp(r_t,2)=1,(-1)^k_t*exp(-(1+k_t)/r_t*I*Pi)),x_t>=0,-1)*w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);

# The "nicer" expression in question
L_phi := (x_t,r_t,k_t,g_t)-> signum(x_t)^(r_t+k_t+1)*abs(g_t)^((k_t+1)/r_t-1)*diff(g_t,x)/r_t;

# The pieces of piecewise phi
phi_l_odd := (x_t,r_t,k_t,g_t)->eval((-1)^k_t*exp(-(1+k_t)/r_t*I*Pi)*w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);
phi_l_even := (x_t,r_t,k_t,g_t)->eval((-1)^k_t*w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);
phi_r := (x_t,r_t,k_t,g_t)->eval(-w^(-(k_t+1)/r_t)/r*exp(-I*w*g_t+(1+k_t)/(2*r_t)*I*Pi)*(GAMMA((1+k_t)/r_t,-I*w*g_t)-GAMMA((1+k_t)/r_t,0)),x=x_t);

# Simplify the basis elements expression with a fixed g(x). This g(x) has a stationary point of order r+1 at x=0. 
# A few notes to anyone playing with this code: 
# try 
# g := x -> -x^r:
# The compact expression doesn't match L(phi) in this case. So it is likely that the assumption eval(diff(g(x),[x$r]),x=0)>0 is used to gain this compact expression.

r := 2:
g := x -> x^r:

assume(w::real):
additionally(w>1):

odd_exp := simplify(L(phi_l_odd(x,r,k,g(x)),g(x)));
even_exp := simplify(L(phi_l_even(x,r,k,g(x)),g(x)));
right_exp := simplify(L(phi_r(x,r,k,g(x)),g(x)));

if(not modp(r,2)=0) then

  for k from 1 to 10 do 
    p[k] := plot(L_phi(x,r,k,g(x)),x=-1..0,color=green,title = "Compact expression" ):
    q[k] := plot(odd_exp,x=-1..0,color=red, title = "L(phi) expression" ):
    b[k] := plot(odd_exp-L_phi(x,r,k,g(x)),x=-1..0,color=red,style = point):
  od:
  plots[display](seq({p[k]},k=1..10));
  plots[display](seq({q[k]},k=1..10));
  plots[display](seq({b[k]},k=1..10));

else

  for k from 1 to 10 do 
    p[k] := plot(L_phi(x,r,k,g(x)),x=-1..0,color=green, title = "Compact Expression" ):
    q[k] := plot(even_exp,x=-1..0,color=red, title = "L(phi) expression" ):
    b[k] := plot(even_exp-L_phi(x,r,k,g(x)),x=-1..0,color=red, style = point , title = "Residual"):
  od:
  plots[display](seq({p[k]},k=1..10));
  plots[display](seq({q[k]},k=1..10));
  plots[display](seq({b[k]},k=1..10));

end if;

for k from 1 to 10 do 
	p[k] := plot(L_phi(x,r,k,g(x)),x=0..1,color=green, title = "Compact Expression" ):
	q[k] := plot(right_exp,x=0..1,color=red, title = "L(phi) expression" ):
	b[k] := plot(right_exp-L_phi(x,r,k,g(x)),x=0..1,color=red, style = point , title = "Residual"):
od:
plots[display](seq({p[k]},k=1..10));
plots[display](seq({q[k]},k=1..10));
plots[display](seq({b[k]},k=1..10));
