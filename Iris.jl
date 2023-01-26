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

export test_iris

function closest_point_to_ellipsoid_ballspace(obstacle, C, d)
    num_vertices = size(obstacle, 1)
    dims = size(obstacle, 2)

    obstacle_ball_space = inv(C) * (obstacle' .- d)

    model = Model(SCS.Optimizer)

    @variable(model, x[1:dims])
    @variable(model, w[1:num_vertices])

    @constraint(model, w in MOI.Nonnegatives(num_vertices))
    @constraint(model, obstacle_ball_space * w .== x)
    @constraint(model, sum(w) == 1)

    @objective(model, Min, x'x)

    optimize!(model)

    value.(x)
end

function closest_point_to_ellipsoid(obstacle, C, d)
    x = closest_point_to_ellipsoid_ballspace(obstacle, C, d)
    C * x + d
end

function closest_obstacles_first!(C, d, obstacles)
    sort!(obstacles, by = obstacle -> norm(
          closest_point_to_ellipsoid_ballspace(obstacle, C, d)))
end

function tangent_plane_to_ellipsoid(x, C, d)
    invC = inv(C)
    a = normalize(invC * invC' * (x - d))
    b = a' * x
    a, b
end

function separating_hyperplanes(C, d, obstacles)
    closest_obstacles_first!(C, d, obstacles)

    num_obstacles = size(obstacles, 1)
    is_significant_obstacle = trues(num_obstacles)

    A = Matrix{Float64}(undef, num_obstacles, 3)
    b = Vector{Float64}(undef, num_obstacles)
    closest_points = Matrix{Float64}(undef, num_obstacles, 3)

    for i in 1:num_obstacles
        if !is_significant_obstacle[i]
            continue
        end

        closest_point = closest_point_to_ellipsoid(obstacles[i], C, d)
        aᵢ, bᵢ = tangent_plane_to_ellipsoid(closest_point, C, d)

        # on the last iteration for i, this loop will be skipped
        for j in i+1:num_obstacles
            is_significant_obstacle[j] = !all(aᵢ' * obstacles[j]' .≥ bᵢ)
        end

        A[i, :] = aᵢ
        b[i] = bᵢ
        closest_points[i, :] = closest_point
    end

    A, b, closest_points, is_significant_obstacle
end

function inscribed_ellipsoid(A, b, Cstart=I(size(A, 2)))
    dims = size(A, 2)

    model = Model(SCS.Optimizer)

    # Start with C as the identity matrix to avoid numerical issues.
    @variable(
        model,
        C[i = 1:dims, j = 1:dims],
        PSD,
        start = Cstart[i, j],
    )

    @variable(model, d[1:dims])

    # ||C * aᵀᵢ||₂ + aᵀᵢ* d ≤ bᵢ ∀ i = [1,...,N]
    @constraint(model, [i = 1:size(A, 1)],
        [b[i] - A[i, :]' * d; C * A[i, :]] in SecondOrderCone())

    # maximize log det C
    @variable(model, logdetC)
    @constraint(model, [logdetC; 1; vec(C)] in MOI.LogDetConeSquare(dims))
    @objective(model, Max, logdetC)

    optimize!(model)

    value.(C), value.(d)
end

function iris(obstacles)
    vis = Visualizer()
    anim = Animation()

    ϵ = .1
    C = ϵ * I(3)
    d = [0., 0., 0.]

    set_obstacles(vis, obstacles)
    set_planes(vis, size(obstacles, 1))
    set_ellipsoid(vis)

    atframe(anim, 0) do
        num_obstacles = size(obstacles, 1)
        draw_obstacles(vis, num_obstacles)
        draw_ellipsoid(vis["ellipsoid"], C, d)
        draw_planes(vis, num_obstacles)
    end

    A = Matrix{Float64}
    b = Vector{Float64}

    tolerance = 1e-3
    max_iterations = 10
    for i in 1:max_iterations
        A, b, closest_points, is_significant =
            separating_hyperplanes(C, d, obstacles)
        C_updated, d = inscribed_ellipsoid(A, b, C)

        atframe(anim, i * 30) do
            draw_ellipsoid(vis["ellipsoid"], C_updated, d)
            draw_planes(vis, A, b, closest_points, is_significant)
        end

        detC = det(C)
        if (det(C_updated) - detC) / detC < tolerance
            println("Finished after ", i, " iterations.")
            break
        end

        C = C_updated
    end

    setanimation!(vis, anim)
    open(vis)
    # MeshCat.convert_frames_to_video(
    #     "/home/chub/Downloads/___________.tar")

    A, b, C, d
end

function test_iris()
    w = 1
    obstacle = [
        0 0 0;
        w 0 0;
        w w 0;
        0 w 0;
        0 0 w;
        w 0 w;
        w w w;
        0 w w;
    ] .- [w/2 w/2 w/2]

    obstacles = [
        obstacle .+ [2 0 0],
        obstacle .+ [0 2 0],
        obstacle .+ [0 0 2],
        obstacle .- [1 0 0],
        obstacle .- [0 1 0],
        obstacle .- [0 0 1],
    ]

    A, b, C, d = iris(obstacles)
end

function set_obstacles(vis, obstacles)
    for (i, obstacle) in enumerate(obstacles)
        set_obstacle(vis["obstacle"][string(i)], obstacle)
    end
end

function set_obstacle(vis, obstacle)
    setobject!(vis, Polyhedra.Mesh(polyhedron(vrep(obstacle))),
               MeshPhongMaterial(color=RGBA(1, 1, 1, 0.5)))
end

function draw_obstacles(vis, num_obstacles)
    for i in 1:num_obstacles
        setvisible!(vis["obstacle"][string(i)], true)
    end
end

function set_ellipsoid(vis)
    setobject!(vis["ellipsoid"], HyperSphere(zero(Point{3, Float64}), 1.0),
               MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5)))
end

function draw_ellipsoid(vis, C, d)
    settransform!(vis, AffineMap(C, d))
end

function set_planes(vis, num_planes)
    for i in 1:num_planes
        set_plane(vis["planes"][string(i)])
        set_closest_point(vis["points"][string(i)])
    end
end

function set_plane(vis)
    xw = .01
    yw = 1.
    zw = 1.
    setobject!(vis, HyperRectangle{3, Float64}([0 0 0], [xw yw zw]),
               MeshPhongMaterial(color=RGBA(0, 1, 0, 0.5)))
end

function draw_planes(vis, num_planes)
    for i in 1:num_planes
        setvisible!(vis["planes"][string(i)], false)
    end
end

function draw_planes(vis, A, b, closest_points, is_significant)
    for i in 1:size(A, 1)
        draw_plane(vis["planes"][string(i)],
                   A[i, :], b[i], closest_points[i, :], is_significant[i])
        draw_closest_point(vis["points"][string(i)],
                           closest_points[i, :], is_significant[i])
    end
end

function draw_plane(vis, a, b, closest_point, shouldshow)
    xw = .01
    yw = 1
    zw = 1
    R = hcat(a, nullspace(a'))
    R[:, end] *= det(R) # make sure this rotation matrix is right handed
    setvisible!(vis, shouldshow)
    settransform!(vis,
        AffineMap(R, closest_point) ∘ Translation(-xw/2, -yw/2, -zw/2))
end

function set_closest_point(vis)
    setobject!(vis, HyperSphere(zero(Point{3, Float64}), .05),
               MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))
end

function draw_closest_point(vis, closest_point, shouldshow)
    setvisible!(vis, shouldshow)
    settransform!(vis, Translation(closest_point))
end

function set_polyhedron(vis)
    setobject!(vis, Polyhedra.Mesh(polyhedron(vrep([[0]]))),
               MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5)))
    setvisible!(vis, false)
end

function draw_polyhedron(vis, A, b)
    poly = HalfSpace(A[1, :], b[1])
    for i = 2:size(A, 1)
        poly = poly ∩ HalfSpace(A[i, :], b[i])
    end
    setobject!(vis, Polyhedra.Mesh(polyhedron(poly)),
               MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5)))
    setvisible!(vis, true)
end

end
