using CSV, Dates, DataFrames, PyCall, TimeZones

# Get working directory
dir = @__DIR__

"""
    fetch_report(; date::Date)

Fetches CPCB AQI PDF bulletin for the given date and converts it to a CSV.
Saves the file to data/YYYY-MM-DD.csv
"""
function fetch_report(; date::Date)
    tabula = pyimport("tabula")

    # Build URL and output file path
    url = "https://cpcb.nic.in//upload/Downloads/AQI_Bulletin_$(replace(string(date), "-" => "")).pdf"
    csv = joinpath(dir, "data", "$date.csv")

    # Try downloading and converting the PDF
    try
        tabula.convert_into(url, csv, lattice=true, output_format="csv", pages="all")
    catch
        if date > today(tz"UTC+05:30") || DateTime(now(tz"UTC+0530")) < DateTime(today(tz"UTC+0530")) + Hour(17)
            error("AQI report for $date not yet available.")
        else
            error("Failed to download or parse AQI PDF.")
        end
    end

    # Read CSV into DataFrame
    df = CSV.read(csv, DataFrame)

    # Clean header
    try select!(df, Not(Symbol("S.No"))) catch end
    headers = [:city, :level, :index, :pollutant, :stations]
    for (n, name) ∈ enumerate(names(df))
        rename!(df, Symbol(name) => headers[n])
    end

    # Clean rows and cells
    deleteat!(df, findall(ismissing, df[:, 3]))
    deleteat!(df, findall(isone, isnothing.(tryparse.(Int, string.(df[:, 3])))))

    for r ∈ 1:nrow(df)
        df[r, 1] = titlecase(replace(replace(df[r, 1], "\r" => " "), "_" => " "))

        plt = ""
        if occursin("3", df[r, 4]) plt *= "O3, " end
        if occursin("Z", df[r, 4]) plt *= "O3, " end
        if occursin("CO", df[r, 4]) plt *= "CO, " end
        if occursin("NO", df[r, 4]) plt *= "NO2, " end
        if occursin("SO", df[r, 4]) plt *= "SO2, " end
        if occursin("10", df[r, 4]) plt *= "PM10, " end
        if occursin("2.5", df[r, 4]) plt *= "PM2.5, " end
        df[r, 4] = rstrip(plt, [',', ' '])

        try df[r, 5] = split(replace(df[r, 5], " #" => ""), "/")[begin] catch end
    end

    # Save cleaned file
    CSV.write(csv, df)
    return
end

# Graceful GitHub Action run
if abspath(PROGRAM_FILE) == @__FILE__
    try
        fetch_report(date=today(tz"UTC+05:30"))
    catch e
        println("🟡 INFO: AQI bulletin not yet available or failed to fetch — skipping update.")
        exit(0)  # Exit gracefully so GitHub Action doesn't fail
    end
end
