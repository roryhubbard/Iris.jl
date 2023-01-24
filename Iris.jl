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

export inscribed_ellipsoid, display_result

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

    vis["polyhedron"]
    setobject!(vis["polyhedron"], feasible_set,
               MeshPhongMaterial(color=RGBA(1, 1, 1, 0.5)))

    vis["ellipsoid"]
    setobject!(vis["ellipsoid"], ellipsoid,
               MeshPhongMaterial(color=RGBA(0, 0, 1, 0.5)))
    settransform!(vis["ellipsoid"], tf)
    
    open(vis)
end

end
