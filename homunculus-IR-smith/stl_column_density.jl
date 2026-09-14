#!/usr/bin/env julia

"""Project a finite-width STL Homunculus shell into a total column-density map.

Usage:
    julia stl_column_density.jl [output_directory] [inclination_deg]

The STL is scaled so its polar radius equals the Smith velocity-law pole at
the 2000 image epoch. A 20-year ejection is represented by 101 homologously
scaled layers whose combined mass is one solar mass.
"""

using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics

const MSUN_G = 1.98847e33
const SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
const KM_TO_CM = 1.0e5
const AU_CM = 1.495978707e13

struct Triangle
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end

struct SampleCurve
    theta::Vector{Float64}
    value::Vector{Float64}
end

function read_binary_stl(path::String)
    open(path, "r") do io
        read(io, 80)
        count = Int(read(io, UInt32))
        triangles = Vector{Triangle}(undef, count)
        for i in 1:count
            read(io, 3 * sizeof(Float32)) # Stored normal; recomputed below.
            a = Float64.(read!(io, Vector{Float32}(undef, 3)))
            b = Float64.(read!(io, Vector{Float32}(undef, 3)))
            c = Float64.(read!(io, Vector{Float32}(undef, 3)))
            read(io, UInt16)
            triangles[i] = Triangle(a, b, c)
        end
        return triangles
    end
end

function read_curve(path::String)
    rows = readdlm(path, ',', String; skipstart = 1)
    values = [(parse(Float64, strip(row[1])), parse(Float64, strip(row[2]))) for row in eachrow(rows) if !isempty(strip(row[1]))]
    return SampleCurve(first.(values), last.(values))
end

"""Shape-preserving cubic Hermite interpolation (PCHIP/Fritsch--Carlson)."""
function smooth_interpolate(x::Vector{Float64}, y::Vector{Float64}, q::Float64)
    q <= x[1] && return y[1]
    q >= x[end] && return y[end]
    i = searchsortedlast(x, q)
    h = diff(x)
    delta = diff(y) ./ h
    slope(k) = begin
        if k == 1
            candidate = ((2h[1] + h[2]) * delta[1] - h[1] * delta[2]) / (h[1] + h[2])
            return sign(candidate) != sign(delta[1]) ? 0.0 : candidate
        elseif k == length(x)
            candidate = ((2h[end] + h[end - 1]) * delta[end] - h[end] * delta[end - 1]) / (h[end] + h[end - 1])
            return sign(candidate) != sign(delta[end]) ? 0.0 : candidate
        elseif delta[k - 1] * delta[k] <= 0
            return 0.0
        end
        w1, w2 = 2h[k] + h[k - 1], h[k] + 2h[k - 1]
        return (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    end
    t = (q - x[i]) / h[i]
    return (2t^3 - 3t^2 + 1) * y[i] + (t^3 - 2t^2 + t) * h[i] * slope(i) +
           (-2t^3 + 3t^2) * y[i + 1] + (t^3 - t^2) * h[i] * slope(i + 1)
end

function curve_at_latitude(curve::SampleCurve, latitude_rad::Float64)
    return smooth_interpolate(deg2rad.(curve.theta), curve.value, clamp(abs(latitude_rad), 0.0, pi / 2))
end

triangle_area(t::Triangle) = norm(cross(t.b - t.a, t.c - t.a)) / 2
triangle_centroid(t::Triangle) = (t.a + t.b + t.c) / 3

function triangle_weight(t::Triangle, mass_curve::SampleCurve)
    centroid = triangle_centroid(t)
    radius = norm(centroid)
    radius == 0 && return 0.0
    normal = cross(t.b - t.a, t.c - t.a)
    normal_norm = norm(normal)
    normal_norm == 0 && return 0.0
    rhat = centroid / radius
    nhat = normal / normal_norm
    latitude = asin(clamp(centroid[3] / radius, -1.0, 1.0))
    domega = triangle_area(t) * abs(dot(rhat, nhat)) / radius^2
    return curve_at_latitude(mass_curve, latitude) * domega
end

function colour(value::Float64, low::Float64, high::Float64)
    t = clamp((log10(value) - log10(low)) / (log10(high) - log10(low)), 0.0, 1.0)
    return round(Int, 255clamp(1.5t, 0, 1)), round(Int, 255clamp(1.5 - 1.5abs(2t - 1), 0, 1)), round(Int, 255clamp(1.5(1 - t), 0, 1))
end

function write_ppm(path::String, image::Matrix{Float64})
    positive = filter(>(0.0), vec(image))
    low, high = quantile(positive, 0.02), quantile(positive, 0.995)
    open(path, "w") do io
        println(io, "P3\n$(size(image, 2)) $(size(image, 1))\n255")
        for row in axes(image, 1), col in axes(image, 2)
            if image[row, col] <= 0
                println(io, "255 255 255")
            else
                r, g, b = colour(image[row, col], low, high)
                println(io, "$r $g $b")
            end
        end
    end
    return low, high
end

function write_colourbar(path::String, low::Float64, high::Float64)
    width, height = 72, 700
    ticks = sort(unique(vcat(log10(low), [t for t in -4.0:0.5:0.0 if log10(low) + 0.05 < t < log10(high) - 0.05], log10(high))))
    open(path, "w") do io
        println(io, "P3\n$width $height\n255")
        for y in 1:height, x in 1:width
            value = 10^(log10(high) - (y - 1) / (height - 1) * (log10(high) - log10(low)))
            r, g, b = colour(value, low, high)
            if x <= 14 && any(abs(y - round(Int, 1 + (log10(high) - tick) / (log10(high) - log10(low)) * (height - 1))) <= 1 for tick in ticks)
                r, g, b = 0, 0, 0
            end
            println(io, "$r $g $b")
        end
    end
end

function rasterize_mass!(image, t::Triangle, triangle_mass, screen_x, screen_y, extent, pixel_area)
    pixels = size(image, 1)
    project(p) = ((dot(p, screen_x) + extent) / (2extent) * pixels,
                  (extent - dot(p, screen_y)) / (2extent) * pixels)
    a, b, c = project(t.a), project(t.b), project(t.c)
    denominator = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
    abs(denominator) < 1e-12 && return
    xmin = max(1, floor(Int, min(a[1], b[1], c[1]) + 0.5))
    xmax = min(pixels, ceil(Int, max(a[1], b[1], c[1]) + 0.5))
    ymin = max(1, floor(Int, min(a[2], b[2], c[2]) + 0.5))
    ymax = min(pixels, ceil(Int, max(a[2], b[2], c[2]) + 0.5))
    covered = Tuple{Int,Int}[]
    for row in ymin:ymax, col in xmin:xmax
        x, y = col - 0.5, row - 0.5
        w1 = ((b[2] - c[2]) * (x - c[1]) + (c[1] - b[1]) * (y - c[2])) / denominator
        w2 = ((c[2] - a[2]) * (x - c[1]) + (a[1] - c[1]) * (y - c[2])) / denominator
        w3 = 1 - w1 - w2
        w1 >= -1e-10 && w2 >= -1e-10 && w3 >= -1e-10 && push!(covered, (row, col))
    end
    if isempty(covered)
        centroid = triangle_centroid(t)
        col = clamp(floor(Int, (dot(centroid, screen_x) + extent) / (2extent) * pixels) + 1, 1, pixels)
        row = clamp(floor(Int, (extent - dot(centroid, screen_y)) / (2extent) * pixels) + 1, 1, pixels)
        image[row, col] += triangle_mass / pixel_area
    else
        contribution = triangle_mass / (length(covered) * pixel_area)
        for (row, col) in covered
            image[row, col] += contribution
        end
    end
end

function main(output_dir::String, inclination_deg::Float64)
    mkpath(output_dir)
    stl_path = joinpath(@__DIR__, "output", "Eta_Car_Homunuculus_model.stl")
    native = read_binary_stl(stl_path)
    mass_curve = read_curve(joinpath(@__DIR__, "Mass-per-SolidAngle.csv"))
    velocity_curve = read_curve(joinpath(@__DIR__, "v_vs_theta.csv"))

    mean_age = 2000.0 - 1847.1
    duration = 20.0
    layers = 101
    smith_pole_radius = curve_at_latitude(velocity_curve, pi / 2) * KM_TO_CM * mean_age * SECONDS_PER_YEAR
    native_pole_radius = maximum(abs(vertex[3]) for t in native for vertex in (t.a, t.b, t.c))
    physical_scale = smith_pole_radius / native_pole_radius
    mean_mesh = [Triangle(physical_scale .* t.a, physical_scale .* t.b, physical_scale .* t.c) for t in native]

    weights = triangle_weight.(mean_mesh, Ref(mass_curve))
    weight_sum = sum(weights)
    mass_fractions = weights ./ weight_sum

    inclination = deg2rad(inclination_deg)
    los = [sin(inclination), 0.0, cos(inclination)]
    screen_x = [0.0, 1.0, 0.0]
    screen_y = normalize(cross(los, screen_x))
    maximum_age = mean_age + duration / 2
    extent = 1.05maximum(max(abs(dot(vertex .* (maximum_age / mean_age), screen_x)), abs(dot(vertex .* (maximum_age / mean_age), screen_y))) for t in mean_mesh for vertex in (t.a, t.b, t.c))
    pixels = 600
    pixel_area = (2extent / pixels)^2
    image = zeros(Float64, pixels, pixels)

    for layer in 1:layers
        launch_offset = ((layer - 0.5) / layers - 0.5) * duration
        radial_factor = (mean_age - launch_offset) / mean_age
        layer_mass = MSUN_G / layers
        for (index, t) in enumerate(mean_mesh)
            scaled = Triangle(radial_factor .* t.a, radial_factor .* t.b, radial_factor .* t.c)
            rasterize_mass!(image, scaled, layer_mass * mass_fractions[index], screen_x, screen_y, extent, pixel_area)
        end
    end

    ppm_path = joinpath(output_dir, "stl_finite_duration_column_density.ppm")
    low, high = write_ppm(ppm_path, image)
    write_colourbar(joinpath(output_dir, "stl_finite_duration_colourbar.ppm"), low, high)
    writedlm(joinpath(output_dir, "stl_finite_duration_column_density_g_cm2.csv"), image, ',')
    sky_axis_au = [(-extent + (index - 0.5) * 2extent / pixels) / AU_CM for index in 1:pixels]
    writedlm(joinpath(output_dir, "stl_sky_axis_au.csv"), sky_axis_au, ',')
    recovered_mass = sum(image) * pixel_area
    open(joinpath(output_dir, "stl_model_metadata.txt"), "w") do io
        println(io, "stl_triangles = $(length(native))")
        println(io, "symmetry_axis = z")
        println(io, "inclination_from_los_deg = $inclination_deg")
        println(io, "total_mass_msun = 1.0")
        println(io, "ejection_epoch_midpoint = 1847.1")
        println(io, "image_epoch = 2000.0")
        println(io, "ejection_duration_years = $duration")
        println(io, "finite_duration_layers = $layers")
        println(io, "native_pole_radius = $native_pole_radius")
        println(io, "smith_pole_radius_cm = $smith_pole_radius")
        println(io, "smith_pole_radius_au = $(smith_pole_radius / AU_CM)")
        println(io, "stl_to_cm_scale = $physical_scale")
        println(io, "map_half_width_cm = $extent")
        println(io, "display_log10_low_g_cm2 = $(log10(low))")
        println(io, "display_log10_high_g_cm2 = $(log10(high))")
        println(io, "recovered_projected_mass_msun = $(recovered_mass / MSUN_G)")
    end
    @printf("Wrote STL finite-duration map to %s (mass recovery %.8f Msun)\n", output_dir, recovered_mass / MSUN_G)
end

main(isempty(ARGS) ? joinpath(@__DIR__, "output") : ARGS[1], length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 41.0)
