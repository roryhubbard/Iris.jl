module Iris

using LinearAlgebra

# optimization
using JuMP
using SCS

# visualization
using ColorTypes
using CoordinateTransformations
using GeometryBasics
using MeshCat
using Polyhedra

export display_result

function closest_point_to_ellipsoid(v, C, d)
    num_vertices = size(v, 1)
    dims = size(v, 2)

    # v is the obstacle vertices in ellipse space
    v_ball_space = inv(C) * (v' .- d)

    model = Model(SCS.Optimizer)

    @variable(model, x[1:dims])
    @variable(model, w[1:num_vertices])

    @constraint(model, w in MOI.Nonnegatives(num_vertices))
    @constraint(model, v_ball_space * w .== x)
    @constraint(model, sum(w) == 1)

    @objective(model, Min, x' * x)

    println(model)

    optimize!(model)

    C * value.(x) + d
end

function tangent_plane_to_ellipsoid(x, C, d)
    invC = inv(C)
    a = normalize(2 * invC * invC' * (x - d))
    b = a' * x
    a, b
end

function inscribed_ellipsoid(A, b)
    dims = size(A, 2)

    model = Model(SCS.Optimizer)

    # Start with C as the identity matrix to avoid numerical issues.
    @variable(
        model,
        C[i = 1:dims, j = 1:dims],
        PSD,
        start = (i == j ? 1.0 : 0.0),
    )

    @variable(model, d[1:dims])

    # ||C * aᵀᵢ||₂ + aᵀᵢ* d ≤ bᵢ ∀ i = [1,...,N]
    @constraint(model, [i = 1:size(A, 1)],
        [b[i] - A[i, :]' * d; C * A[i, :]] in SecondOrderCone())

    # maximize log det C
    @variable(model, logdetC)
    @constraint(model, [logdetC; 1; vec(C)] in MOI.LogDetConeSquare(dims))
    @objective(model, Max, logdetC)

    println(model)

    optimize!(model)

    value.(C), value.(d)
end

function display_result()
    A = [ 0 -1  0;
          1  0  0;
          0  1  0;
         -1  0  0;
          0  0 -1;
          0  0  1]

    b = [0 1 1 0 0 2]'

    poly = HalfSpace(A[1, :], b[1])
    for i = 2:size(A, 1)
        poly = poly ∩ HalfSpace(A[i, :], b[i])
    end
    feasible_set = Polyhedra.Mesh(polyhedron(poly))

    C, d = inscribed_ellipsoid(A, b)

    ellipsoid = HyperSphere(zero(Point{3, Float64}), 1.0)
    tf = AffineMap(C, d)

    vis = Visualizer()

    setobject!(vis["polyhedron"], feasible_set,
               MeshPhongMaterial(color=RGBA(1, 1, 1, 0.5)))

    setobject!(vis["ellipsoid"], ellipsoid,
               MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5)))
    settransform!(vis["ellipsoid"], tf)

    obstacle = [
        2 2 2;
        3 2 2;
        3 3 2;
        2 3 2;
        2 2 3;
        3 2 3;
        3 3 3;
        2 3 3;
    ]

    closest_point = closest_point_to_ellipsoid(obstacle, C, d)

    obstacle_viz = Polyhedra.Mesh(polyhedron(vrep(obstacle)))
    setobject!(vis["obstacle"], obstacle_viz,
               MeshPhongMaterial(color=RGBA(1, 1, 1, 0.5)))

    x_ellipse_space_viz = HyperSphere(Point{3, Float64}(closest_point), .05)
    setobject!(vis["closest_point"], x_ellipse_space_viz,
               MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))

    a, b = tangent_plane_to_ellipsoid(closest_point, C, d)

    draw_plane(vis, a, b, closest_point)
    
    open(vis)
end

function draw_plane(vis, a, b, x)
    R = hcat(a, nullspace(a'))

    xw = .01
    yw = 1
    zw = 1

    plane = HyperRectangle{3, Float64}([0 0 0], [xw yw zw])
    tf = AffineMap(R, x) ∘ Translation(-xw/2, -yw/2, -zw/2)

    setobject!(vis["plane"], plane, MeshPhongMaterial(color=RGBA(0, 1, 0, 0.5)))
    settransform!(vis["plane"], tf)
end

end
