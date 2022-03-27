Digits := 16:

# Setup the problem
f := x -> 1:
g := x -> -(x-0.1)^2:
a := 0:
b := Pi:
w_range := [seq(k,k=10..100,1)]:

# Finding all the stationary points
stat_points_set := {(solve(diff(g(x),x),x))}:

stat_points := []:
int_limits := []:

# Separating the points that lie within [a,b] from the rest
for i from 1 to nops(stat_points_set) do
    if(normal(Im(stat_points_set[i]))=0) then
        if(evalf(stat_points_set[i]-b)<0 and evalf(stat_points_set[i]-a)>0) then
        stat_points := [op(stat_points), stat_points_set[i]]
        end if:
    end if:
od:
stat_points := sort(evalf(stat_points)):

# Finding the order of each of the stationary points
orders := [seq(2,i=1..nops(stat_points))]:
for i from 1 to nops(stat_points) do 
    for j from 2 to 15 do
        if(eval(diff(g(x),[x$j]),x=stat_points[i])=0) then
            orders[i] := orders[i]+1:
        else
            break:
        end if:
    od:
od:

# Checking whether the second derivative vanishes away from the stationary points
second_deriv_zero_set := [op({solve(diff(g(x),[x$2])=0)})]:
real_second_deriv_zero_set := []:

if(second_deriv_zero_set = []) then
    # If the second derivative vanishes everywhere in [a,b], continue
    real_second_deriv_zero_set := []:
else
    for i from 1 to nops(second_deriv_zero_set) do
    # use evalf on any sols that have a RootOf
    if(type(second_deriv_zero_set[i],RootOf)) then
        second_deriv_zero_set[i] := evalf(allvalues(second_deriv_zero_set[i])):
    end if:
        # Ignore all the points which have a non-zero complex part
       if(Im(second_deriv_zero_set[i])=0) then
            # Add a point to the list if the second derivative vanishes but the first doesn't 
            if(not(member(second_deriv_zero_set[i],stat_points))) then
                if(evalf(second_deriv_zero_set[i])>a and evalf(second_deriv_zero_set[i])<b) then
                    real_second_deriv_zero_set := [op(real_second_deriv_zero_set), second_deriv_zero_set[i]]:
                end if:
            end if:
       end if:
    od:
end if:
real_second_deriv_zero_set := sort(real_second_deriv_zero_set):

# If there are no stationary points in the interval, use Levin-Hermite on the entire interval.
if(nops(stat_points)=0) then 
    int_limits := []:
else
    # For each stationary point, make a list of 
    # 1. stationary points to its left
    # 2. stationary points to its right
    # 3. and 4. The same but for points where the second derivative vanishes
    for i from 1 to nops(stat_points) do
        Left_stat_point_set := []:
        Right_stat_point_set := []:
        Left_second_deriv_zero_set := []:
        Right_second_deriv_zero_set := []:
        Left_interesting_points := []:
        Right_interesting_points := []:

	
        for j from 1 to nops(real_second_deriv_zero_set) do
            if(not normal(real_second_deriv_zero_set[j]) = normal(stat_points[i])) then
                if(normal(real_second_deriv_zero_set[j])<normal(stat_points[i])) then
                    Left_second_deriv_zero_set := [op(Left_second_deriv_zero_set),real_second_deriv_zero_set[j]]:
                else
                    Right_second_deriv_zero_set := [op(Right_second_deriv_zero_set),real_second_deriv_zero_set[j]]:
                end if:
            end if:
        od:

        for j from 1 to nops(stat_points) do
            if(not (i=j)) then              
                if(normal(stat_points[j])<stat_points[i]) then
                    Left_stat_point_set := [op(Left_stat_point_set),stat_points[j]]:
                else 
                    Right_stat_point_set := [op(Right_stat_point_set),stat_points[j]]:
                end if:
            end if:
        od:

        Left_second_deriv_zero_set := convert(sort(Left_second_deriv_zero_set),float):
        Right_second_deriv_zero_set := convert(sort(Right_second_deriv_zero_set),float):
        Left_stat_point_set := convert(sort(Left_stat_point_set),float):
        Right_stat_point_set := convert(sort(Right_stat_point_set),float):
	
	# Concatenate the lists into two lists as follows: 
	# 1. stationary points on the left + points where second derivative vanishes to the left
	# 2. Same but for the right side
        Left_interesting_points := {op(Left_second_deriv_zero_set),op(Left_stat_point_set)}:
        Right_interesting_points := {op(Right_second_deriv_zero_set),op(Right_stat_point_set)}:

	# Take a small enough symmetric interval around each stationary point which excludes all the points in the list
        dist_left := 0:
        dist_right := 0: 
        if(Left_interesting_points = {}) then
            Lower_limit := a:
            dist_left := abs(stat_points[i]-a):
        else
            Lower_limit := (max(Left_interesting_points)+stat_points[i])/2:
            dist_left := abs(stat_points[i]-max(Left_interesting_points)):
        end if:

        if(Right_interesting_points = {}) then
            Upper_limit := b:
            dist_right := abs(stat_points[i]-b):
        else
            Upper_limit := (stat_points[i]+min(Right_interesting_points))/2:
            dist_right := abs(stat_points[i]-min(Right_interesting_points)):
        end if:

        radius_interval := min(dist_right,dist_left)/2:
        int_limits := [op(int_limits),[stat_points[i]-radius_interval,stat_points[i]+radius_interval]]:
    od:
end if:

# Make a list of intervals which have not already been included
levin_int_limits := []:
if(int_limits = []) then
    levin_int_limits := [[a,b]]:
else
    if( not int_limits[1][1] = a ) then 
        levin_int_limits := [[a,int_limits[1][1]]]:
    end if:
    
    for i from 1 to nops(int_limits)-1 do 
        if(not int_limits[i][2] = int_limits[i+1][1]) then
            levin_int_limits := [op(levin_int_limits),[int_limits[i][2],int_limits[i+1][1]]]:
        end if:
    od:

    if(not int_limits[nops(int_limits)][2] = b) then 
        levin_int_limits := [op(levin_int_limits),[int_limits[nops(int_limits)][2],b]]:
    end if:
end if: 


printf("The stationary points which lie in the integration domain are \n"):
print(stat_points):
printf("The orders of the stationary points are \n"):
print(orders):
printf("Integration limits over with Moment free filon needs to be used are:\n"):
print(int_limits):
printf("Integration limits over with Levin integration is to be used are:\n"):
print(levin_int_limits):

read "Levin_Hermite_Maple.mpl":
read "Moment_free_Filon.mpl":
read "filon_stat_point_proc.mpl":

int_g := [seq(0,k=1..nops(w_range))]:


for i from 1 to nops(stat_points) do
    print("Filon Method Called"):
    if(not evalf(g(stat_points[i]))=0) then
        st_pt := stat_points[i]:
        g_norm := x -> g(x)-g(st_pt):
            if(diff(g(x),[x$r])<0) then
                temp := Filon_Method_Stat_Pts(f,g_norm,int_limits[i][1],int_limits[i][2],5,2,stat_points[i],orders[i]-1,1,w_range):
            else 
                temp := Moment_free_filon(f,g_norm,int_limits[i],stat_points[i],orders[i],2,5,w_range):
            end if:
        int_g := int_g + [seq(temp[k]*exp(I*w_range[k]*g(st_pt)),k=1..nops(w_range))]:
    else
        if(diff(g(x),[x$r])<0) then
            int_g := int_g + Filon_Method_Stat_Pts(f,g,int_limits[i][1],int_limits[i][2],5,2,stat_points[i],orders[i]-1,1,w_range):
        else
            int_g := int_g + Moment_free_filon(f,g,int_limits[i],stat_points[i],orders[i],2,4,w_range);
        end if:
    end if:
od:

for i from 1 to nops(levin_int_limits) do
    int_g := int_g + Levin_Hermite(f,g,levin_int_limits[i],5,2,w_range):
od:
