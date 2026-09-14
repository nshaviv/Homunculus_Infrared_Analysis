#!/usr/bin/env julia

"""Render an axisymmetric, infinitesimally thin Homunculus shell.

Usage:
    julia thin_shell_column_density.jl [output_directory]

Input files are read from the directory containing this script.  The output
map is a physical gas surface density in g cm^-2 for a total shell mass of
one solar mass.  The projection includes all visible surface crossings.
"""

using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics

const MSUN_G = 1.98847e33
const SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
const KM_TO_CM = 1.0e5

struct SampleCurve
    theta::Vector{Float64}
    value::Vector{Float64}
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
            return sign(candidate) != sign(delta[1]) ? 0.0 : (sign(delta[1]) != sign(delta[2]) && abs(candidate) > abs(3delta[1]) ? 3delta[1] : candidate)
        elseif k == length(x)
            candidate = ((2h[end] + h[end - 1]) * delta[end] - h[end] * delta[end - 1]) / (h[end] + h[end - 1])
            return sign(candidate) != sign(delta[end]) ? 0.0 : (sign(delta[end]) != sign(delta[end - 1]) && abs(candidate) > abs(3delta[end]) ? 3delta[end] : candidate)
        elseif delta[k - 1] * delta[k] <= 0
            return 0.0
        else
            w1, w2 = 2h[k] + h[k - 1], h[k] + 2h[k - 1]
            return (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        end
    end
    t = (q - x[i]) / h[i]
    h00, h10 = 2t^3 - 3t^2 + 1, t^3 - 2t^2 + t
    h01, h11 = -2t^3 + 3t^2, t^3 - t^2
    return h00 * y[i] + h10 * h[i] * slope(i) + h01 * y[i + 1] + h11 * h[i] * slope(i + 1)
end

"""Interpolate supplied latitude-from-equator data on a polar-angle surface."""
function axisymmetric_value(curve::SampleCurve, theta::Float64)
    latitude = abs(pi / 2 - theta)
    return smooth_interpolate(deg2rad.(curve.theta), curve.value, latitude)
end

function normalized_mass_distribution(curve::SampleCurve)
    # Integral of the mirrored distribution over 4pi steradians.
    theta = range(0.0, pi / 2; length = 10_001)
    integral = 4pi * sum(0.5 * (axisymmetric_value(curve, theta[i]) * sin(theta[i]) + axisymmetric_value(curve, theta[i + 1]) * sin(theta[i + 1])) * (theta[i + 1] - theta[i]) for i in 1:length(theta)-1)
    return theta -> axisymmetric_value(curve, theta) / integral
end

function triangle_normal(a, b, c)
    n = cross(b - a, c - a)
    return n / norm(n)
end

function colour(value::Float64, low::Float64, high::Float64)
    t = clamp((log10(value) - log10(low)) / (log10(high) - log10(low)), 0.0, 1.0)
    return round(Int, 255 * clamp(1.5t, 0, 1)), round(Int, 255 * clamp(1.5 - abs(2t - 1) * 1.5, 0, 1)), round(Int, 255 * clamp(1.5 * (1 - t), 0, 1))
end

function write_ppm(path::String, image::Matrix{Float64})
    positive = filter(>(0.0), vec(image))
    low, high = quantile(positive, 0.02), quantile(positive, 0.995)
    open(path, "w") do io
        println(io, "P3")
        println(io, "$(size(image, 2)) $(size(image, 1))")
        println(io, "255")
        for row in axes(image, 1), col in axes(image, 2)
            value = image[row, col]
            if value <= 0
                println(io, "255 255 255")
            else
                r, g, b = colour(value, low, high)
                println(io, "$r $g $b")
            end
        end
    end
    return low, high
end

function write_colourbar(path::String, low::Float64, high::Float64, ticks::Vector{Float64})
    width, height = 72, 700
    open(path, "w") do io
        println(io, "P3")
        println(io, "$width $height")
        println(io, "255")
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

function main(output_dir::String)
    source_dir = @__DIR__
    mkpath(output_dir)
    mass_curve = read_curve(joinpath(source_dir, "Mass-per-SolidAngle.csv"))
    velocity_curve = read_curve(joinpath(source_dir, "v_vs_theta.csv"))
    mass_per_sr = normalized_mass_distribution(mass_curve)

    # Smith (2006): symmetry axis is 41 degrees from the line of sight.
    inclination = deg2rad(41.0)
    los = [sin(inclination), 0.0, cos(inclination)]
    screen_x = [0.0, 1.0, 0.0]
    screen_y = normalize(cross(los, screen_x))
    thin_age_years = 2000.0 - 1847.1
    ejection_duration_years = 20.0
    layer_count = 101
    equatorial_latitude_cutoff = deg2rad(10.0)

    n_theta, n_phi, pixels = 181, 361, 600
    theta_grid = collect(range(0.0, pi; length = n_theta))
    phi_grid = collect(range(0.0, 2pi; length = n_phi))
    radius(theta, age_years) = axisymmetric_value(velocity_curve, theta) * KM_TO_CM * age_years * SECONDS_PER_YEAR
    max_radius = maximum(radius(theta, thin_age_years + ejection_duration_years / 2) for theta in theta_grid)
    extent = 1.05 * max_radius
    image = zeros(Float64, pixels, pixels)
    duration_image = zeros(Float64, pixels, pixels)
    near_image = zeros(Float64, pixels, pixels)
    near_duration_image = zeros(Float64, pixels, pixels)

    # Rasterise each surface triangle once.  Contributions add along a pixel's
    # ray, so near/far sides and any additional crossings are retained.
    function rasterize(a, b, c, theta_center, mass_fraction, target, near_target, near_depth)
        near_side_allowed = abs(pi / 2 - theta_center) >= equatorial_latitude_cutoff
        pa = [dot(a, screen_x), dot(a, screen_y)]
        pb = [dot(b, screen_x), dot(b, screen_y)]
        pc = [dot(c, screen_x), dot(c, screen_y)]
        to_pixel(p) = ((p[1] / extent + 1) * (pixels - 1) / 2 + 1, (1 - p[2] / extent) * (pixels - 1) / 2 + 1)
        ua, ub, uc = to_pixel(pa), to_pixel(pb), to_pixel(pc)
        denominator = (ub[2] - uc[2]) * (ua[1] - uc[1]) + (uc[1] - ub[1]) * (ua[2] - uc[2])
        abs(denominator) < eps() && return
        normal = triangle_normal(a, b, c)
        # Surface mass density follows dM = Sigma dA and dA/dOmega = r^2 / |rhat dot n|.
        centroid = (a + b + c) / 3
        rhat = centroid / norm(centroid)
        # The mesh winding is not guaranteed to be outward at both poles.
        dot(normal, rhat) < 0 && (normal = -normal)
        sigma_surface = mass_fraction * MSUN_G * mass_per_sr(theta_center) * abs(dot(rhat, normal)) / norm(centroid)^2
        contribution = sigma_surface / max(abs(dot(normal, los)), 1e-8)
        surface_depth = dot(centroid, los)
        xmin = max(1, floor(Int, min(ua[1], ub[1], uc[1])))
        xmax = min(pixels, ceil(Int, max(ua[1], ub[1], uc[1])))
        ymin = max(1, floor(Int, min(ua[2], ub[2], uc[2])))
        ymax = min(pixels, ceil(Int, max(ua[2], ub[2], uc[2])))
        for y in ymin:ymax, x in xmin:xmax
            w1 = ((ub[2] - uc[2]) * (x - uc[1]) + (uc[1] - ub[1]) * (y - uc[2])) / denominator
            w2 = ((uc[2] - ua[2]) * (x - uc[1]) + (ua[1] - uc[1]) * (y - uc[2])) / denominator
            w3 = 1 - w1 - w2
            if w1 >= -1e-10 && w2 >= -1e-10 && w3 >= -1e-10
                target[y, x] += contribution
                # Retain only the first observer-facing intersection for this
                # ballistic layer. This prevents folded inner surfaces from
                # appearing through the outer Homunculus wall.
                if near_side_allowed && dot(normal, los) > 0 && surface_depth > near_depth[y, x]
                    near_target[y, x] = contribution
                    near_depth[y, x] = surface_depth
                end
            end
        end
    end

    function add_ballistic_layer!(target, near_target, age_years, mass_fraction)
        layer_near = zeros(Float64, pixels, pixels)
        layer_near_depth = fill(-Inf, pixels, pixels)
        vertex(theta, phi) = radius(theta, age_years) .* [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
        for i in 1:n_theta-1, j in 1:n_phi-1
            a, b = vertex(theta_grid[i], phi_grid[j]), vertex(theta_grid[i + 1], phi_grid[j])
            c, d = vertex(theta_grid[i + 1], phi_grid[j + 1]), vertex(theta_grid[i], phi_grid[j + 1])
            theta_center = (theta_grid[i] + theta_grid[i + 1]) / 2
            rasterize(a, b, c, theta_center, mass_fraction, target, layer_near, layer_near_depth)
            rasterize(a, c, d, theta_center, mass_fraction, target, layer_near, layer_near_depth)
        end
        near_target .+= layer_near
    end
    add_ballistic_layer!(image, near_image, thin_age_years, 1.0)

    # Uniform mass loss over 20 yr, sampled at midpoint launch times around
    # the measured mean ejection epoch. Each layer retains ballistic geometry.
    for layer in 1:layer_count
        offset = ((layer - 0.5) / layer_count - 0.5) * ejection_duration_years
        add_ballistic_layer!(duration_image, near_duration_image, thin_age_years - offset, 1.0 / layer_count)
    end

    ppm_path = joinpath(output_dir, "thin_shell_column_density.ppm")
    low, high = write_ppm(ppm_path, image)
    tick_values = sort(unique(vcat(log10(low), [tick for tick in [-2.0, -1.5, -1.0] if log10(low) + 0.05 < tick < log10(high) - 0.05], log10(high))))
    write_colourbar(joinpath(output_dir, "thin_shell_column_density_colourbar.ppm"), low, high, tick_values)
    writedlm(joinpath(output_dir, "thin_shell_column_density_g_cm2.csv"), image, ',')
    near_ppm_path = joinpath(output_dir, "near_side_thin_shell_column_density.ppm")
    near_low, near_high = write_ppm(near_ppm_path, near_image)
    writedlm(joinpath(output_dir, "near_side_thin_shell_column_density_g_cm2.csv"), near_image, ',')
    duration_ppm_path = joinpath(output_dir, "finite_duration_column_density.ppm")
    duration_low, duration_high = write_ppm(duration_ppm_path, duration_image)
    duration_ticks = sort(unique(vcat(log10(duration_low), [tick for tick in [-2.0, -1.5] if log10(duration_low) + 0.05 < tick < log10(duration_high) - 0.05], log10(duration_high))))
    write_colourbar(joinpath(output_dir, "finite_duration_column_density_colourbar.ppm"), duration_low, duration_high, duration_ticks)
    writedlm(joinpath(output_dir, "finite_duration_column_density_g_cm2.csv"), duration_image, ',')
    near_duration_ppm_path = joinpath(output_dir, "near_side_finite_duration_column_density.ppm")
    near_duration_low, near_duration_high = write_ppm(near_duration_ppm_path, near_duration_image)
    near_duration_ticks = sort(unique(vcat(log10(near_duration_low), [tick for tick in [-2.5, -2.0, -1.5] if log10(near_duration_low) + 0.05 < tick < log10(near_duration_high) - 0.05], log10(near_duration_high))))
    write_colourbar(joinpath(output_dir, "near_side_finite_duration_colourbar.ppm"), near_duration_low, near_duration_high, near_duration_ticks)
    writedlm(joinpath(output_dir, "near_side_finite_duration_column_density_g_cm2.csv"), near_duration_image, ',')
    open(joinpath(output_dir, "model_metadata.txt"), "w") do io
        println(io, "total_mass_msun = 1.0")
        println(io, "ejection_epoch = 1847.1")
        println(io, "image_epoch = 2000.0")
        println(io, "axis_inclination_from_los_deg = 41.0")
        println(io, "map_half_width_cm = $extent")
        println(io, "map_half_width_au = $(extent / 1.495978707e13)")
        println(io, "display_log10_low_g_cm2 = $(log10(low))")
        println(io, "display_log10_high_g_cm2 = $(log10(high))")
        println(io, "ejection_duration_years = $ejection_duration_years")
        println(io, "finite_duration_layers = $layer_count")
        println(io, "equatorial_latitude_cutoff_deg = $(rad2deg(equatorial_latitude_cutoff))")
        println(io, "finite_duration_display_log10_low_g_cm2 = $(log10(duration_low))")
        println(io, "finite_duration_display_log10_high_g_cm2 = $(log10(duration_high))")
        println(io, "near_side_thin_display_log10_low_g_cm2 = $(log10(near_low))")
        println(io, "near_side_thin_display_log10_high_g_cm2 = $(log10(near_high))")
        println(io, "near_side_finite_duration_display_log10_low_g_cm2 = $(log10(near_duration_low))")
        println(io, "near_side_finite_duration_display_log10_high_g_cm2 = $(log10(near_duration_high))")
    end
    @printf("Wrote %s and map data to %s\n", ppm_path, output_dir)
end

main(isempty(ARGS) ? joinpath(@__DIR__, "output") : ARGS[1])
