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

export animate_iris, test_iris

function closest_point_to_ellipsoid_ballspace(obstacle, C, d)
    num_vertices = size(obstacle, 1)
    dims = size(obstacle, 2)

    obstacle_ball_space = inv(C) * (obstacle' .- d)

    model = Model(SCS.Optimizer)
    set_silent(model)

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
    a = normalize(2 * invC * invC' * (x - d))
    b = a' * x
    a, b
end

function separating_hyperplanes(obstacles, C, d)
    num_obstacles = size(obstacles, 1)
    issignificant = trues(num_obstacles)

    A = Matrix{Float64}(undef, num_obstacles, 3)
    b = Vector{Float64}(undef, num_obstacles)
    closest_points = Matrix{Float64}(undef, num_obstacles, 3)

    for i in 1:num_obstacles
        if !issignificant[i]
            continue
        end

        closest_point = closest_point_to_ellipsoid(obstacles[i], C, d)
        aᵢ, bᵢ = tangent_plane_to_ellipsoid(closest_point, C, d)

        # on the last iteration for i, this loop will be skipped
        for j in i+1:num_obstacles
            issignificant[j] = !all(aᵢ' * obstacles[j]' .≥ bᵢ)
        end

        A[i, :] = aᵢ
        b[i] = bᵢ
        closest_points[i, :] = closest_point
    end

    A, b, closest_points, issignificant
end

function inscribed_ellipsoid(bounding_polytope, A, b, Cstart=I(size(A, 2)))
    dims = size(A, 2)

    model = Model(SCS.Optimizer)
    set_silent(model)

    # Start with C as the identity matrix to avoid numerical issues.
    @variable(
        model,
        C[i = 1:dims, j = 1:dims],
        PSD,
        start = Cstart[i, j],
    )

    @variable(model, d[1:dims])

    for halfspace in halfspaces(bounding_polytope)
        @constraint(model, [halfspace.β - halfspace.a' * d; C * halfspace.a]
                    in SecondOrderCone())
    end

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

function iris_iteration(bounding_polytope, obstacles, A, b, C, d)
    A, b, closest_points, issignificant =
        separating_hyperplanes(obstacles, C, d)
    C, d = inscribed_ellipsoid(bounding_polytope, A[issignificant, :],
                               b[issignificant], C)
    A, b, closest_points, issignificant, C, d
end

function iris(bounding_polytope, obstacles, starting_point)
    ϵ = .1
    C = ϵ * I(3)
    d = starting_point

    # Not sure if this is worth it. The ellipsoid might transform so much after
    # some iterations that the ordering is highly inaccurate. Moving this sort
    # into `separating_hyperplanes` would make the ordering always accurate but
    # then we are solving the closest points on all obstacles which kind of
    # defeats the purpose of sorting to begin with.
    closest_obstacles_first!(C, d, obstacles)

    A = Matrix{Float64}
    b = Vector{Float64}
    issignificant = Vector{Bool}

    tolerance = 1e-3
    max_iterations = 10
    i = 1
    while i < max_iterations
        A, b, closest_points, issignificant, Cnext, dnext =
            iris_iteration(bounding_polytope, obstacles, A, b, C, d)

        detC = det(C)
        detCnext = det(Cnext)
        C = Cnext
        d = dnext

        if (det(Cnext) - detC) / detC < tolerance
            break
        end

        i += 1
    end
    println("Finished after ", i, " iterations.")

    # The rows of A and b are undefined where is_signficant is false.
    A[issignificant, :], b[issignificant], C, d
end

function iris_with_animation(bounding_polytope,
                             obstacles, starting_point)
    vis = Visualizer()
    anim = Animation()
    framedelta = 30

    ϵ = .1
    C = ϵ * I(3)
    d = starting_point

    # Not sure if this is worth it. The ellipsoid might transform so much after
    # some iterations that the ordering is highly inaccurate. Moving this sort
    # into `separating_hyperplanes` would make the ordering always accurate but
    # then we are solving the closest points on all obstacles which kind of
    # defeats the purpose of sorting to begin with.
    closest_obstacles_first!(C, d, obstacles)

    set_bounding_polytope(vis, bounding_polytope)
    set_obstacles(vis, obstacles)
    set_planes(vis, size(obstacles, 1))
    set_ellipsoid(vis)

    atframe(anim, 0) do
        num_obstacles = size(obstacles, 1)
        draw_obstacles(vis, num_obstacles)
        draw_planes(vis, num_obstacles)
        draw_ellipsoid(vis, C, d)
    end

    A = Matrix{Float64}
    b = Vector{Float64}
    issignificant = Vector{Bool}

    tolerance = 1e-3
    max_iterations = 10
    i = 1
    while i < max_iterations
        A, b, closest_points, issignificant, Cnext, dnext =
            iris_iteration(bounding_polytope, obstacles, A, b, C, d)

        atframe(anim, i * framedelta) do
            draw_planes(vis, A, b, closest_points, issignificant)
            draw_ellipsoid(vis, C, d)
        end

        atframe(anim, (i+1) * framedelta) do
            draw_ellipsoid(vis, Cnext, dnext)
        end

        detC = det(C)
        detCnext = det(Cnext)
        C = Cnext
        d = dnext

        if (det(Cnext) - detC) / detC < tolerance
            break
        end

        i += 1
    end
    println("Finished after ", i, " iterations.")

    # The rows of A and b are undefined where is_signficant is false.
    A = A[issignificant, :]
    b = b[issignificant]

    for halfspace in halfspaces(bounding_polytope)
        A = vcat(A, halfspace.a')
        b = vcat(b, halfspace.β)
    end

    set_free_space_polytope(vis, A, b)
    atframe(anim, 0) do
        setvisible!(vis["free_space_polytope"], false)
    end
    atframe(anim, (i+2) * framedelta) do
        setvisible!(vis["free_space_polytope"], true)
    end

    setanimation!(vis, anim)
    open(vis)
    # MeshCat.convert_frames_to_video(
    #     "/home/chub/Downloads/___________.tar")

    A, b, C, d
end

function animate_iris()
    s = 10.
    boundingbox_A = [
        1   0  0;
        -1  0  0;
        0   1  0;
        0  -1  0;
        0   0  1;
        0   0 -1
    ]
    boundingbox_b = [s, 0, s, 0, s, 0]
    bounding_polytope = translate(hrep(boundingbox_A, boundingbox_b),
                                  -[s/2, s/2, s/2])

    w = 1. obstacle = [
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
#        obstacle .+ [2 0 0],
        obstacle .+ [0 2 0],
        obstacle .+ [0 0 2],
        obstacle .- [1 0 0],
        obstacle .- [0 1 0],
        obstacle .- [0 0 1],
    ]
    starting_point = [0., 0., 0.] # q0 in paper

    A, b, C, d = iris_with_animation(
        bounding_polytope, obstacles, starting_point)
end

function test_iris()
    s = 10.
    boundingbox_A = [
        1   0  0;
        -1  0  0;
        0   1  0;
        0  -1  0;
        0   0  1;
        0   0 -1
    ]
    boundingbox_b = [s, 0, s, 0, s, 0]
    bounding_polytope = translate(hrep(boundingbox_A, boundingbox_b),
                                  -[s/2, s/2, s/2])

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
#        obstacle .+ [2 0 0],
        obstacle .+ [0 2 0],
        obstacle .+ [0 0 2],
        obstacle .- [1 0 0],
        obstacle .- [0 1 0],
        obstacle .- [0 0 1],
    ]
    starting_point = [0., 0., 0.] # q0 in paper

    A, b, C, d = iris(bounding_polytope, obstacles, starting_point)

    println("A: ", A)
    println("b: ", b)
    println("C: ", C)
    println("d: ", d)
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
    settransform!(vis["ellipsoid"], AffineMap(C, d))
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
        setvisible!(vis["points"][string(i)], false)
    end
end

function draw_planes(vis, A, b, closest_points, issignificant)
    for i in 1:size(A, 1)
        draw_plane(vis["planes"][string(i)],
                   A[i, :], b[i], closest_points[i, :], issignificant[i])
        draw_closest_point(vis["points"][string(i)],
                           closest_points[i, :], issignificant[i])
    end
end

function draw_plane(vis, a, b, closest_point, shouldshow)
    setvisible!(vis, shouldshow)
    if !shouldshow
        return
    end

    xw = .01
    yw = 1
    zw = 1

    R = hcat(a, nullspace(a'))
    R[:, end] *= det(R) # make sure this rotation matrix is right handed
    settransform!(vis,
        AffineMap(R, closest_point) ∘ Translation(-xw/2, -yw/2, -zw/2))
end

function set_closest_point(vis)
    setobject!(vis, HyperSphere(zero(Point{3, Float64}), .05),
               MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))
end

function draw_closest_point(vis, closest_point, shouldshow)
    setvisible!(vis, shouldshow)
    if !shouldshow
        return
    end
    settransform!(vis, Translation(closest_point))
end

function set_free_space_polytope(vis, A, b)
    setobject!(vis["free_space_polytope"],
               Polyhedra.Mesh(polyhedron(hrep(A, b))),
               MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5)))
end

function set_bounding_polytope(vis, polytope::HRepresentation)
    setobject!(vis["bounding_polytope"], Polyhedra.Mesh(polyhedron(polytope)),
               MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5)))
    setvisible!(vis["bounding_polytope"], false)
end

end
