# # # # # # # # # # # # # # # # # # # # # # # # # #
#
#  This script fetches the daily Air Quality Index (AQI)
#  bulletin from the CPCB website, processes it, and
#  saves it as a CSV file in the /data directory.
#
#  It is designed to be run by the GitHub Action in this
#  repository.
#
# # # # # # # # # # # # # # # # # # # # # # # # # #

using CSV
using Dates
using DataFrames
using PyCall
using TimeZones

"""
    fetch_and_process_bulletin(date_to_fetch::Date)

Fetches the AQI bulletin for a given date, cleans the data,
and saves it to the `data/` directory.
"""
function fetch_and_process_bulletin(date_to_fetch::Date)
    println("Attempting to fetch AQI bulletin for: ", date_to_fetch)

    # --- 1. Fetch table from CPCB PDF ---
    tabula = pyimport("tabula")
    url = "https://cpcb.nic.in/upload/Downloads/AQI_Bulletin_$(replace(string(date_to_fetch), "-" => "")).pdf"
    output_path = joinpath(dirname(@__DIR__), "data", "$(date_to_fetch).csv")

    try
        # Use tabula to convert the PDF table directly into a CSV file
        tabula.convert_into(url, output_path, lattice=true, output_format="csv", pages="all")
        println("Successfully downloaded and converted PDF from: ", url)
    catch e
        println("Error: Could not fetch or process the PDF from the URL.")
        println("This usually means the bulletin for '$date_to_fetch' is not yet available.")
        println("Underlying error: ", e)
        # Exit the script gracefully if the file doesn't exist.
        # The GitHub Action will continue and report "nothing to commit".
        return
    end

    # --- 2. Read the raw CSV and clean it ---
    df = CSV.read(output_path, DataFrame, silencewarnings=true)

    # Remove the serial number column if it exists
    if "S.No" in names(df)
        select!(df, Not(Symbol("S.No")))
    end

    # Standardize column headers
    headers = [:city, :level, :index, :pollutant, :stations]
    rename!(df, headers, makeunique=true)

    # Remove empty or invalid rows
    deleteat!(df, findall(ismissing, df.index))
    deleteat!(df, findall(x -> tryparse(Int, string(x)) === nothing, df.index))

    # --- 3. Clean data within cells ---
    for r in 1:nrow(df)
        # Clean city name
        df[r, :city] = titlecase(replace(df[r, :city], "\r" => " ", "_" => " "))

        # Standardize pollutant names
        pollutant_str = ""
        p_val = ismissing(df[r, :pollutant]) ? "" : df[r, :pollutant]
        if occursin("3", p_val) || occursin("Z", p_val) pollutant_str *= "O3, " end
        if occursin("CO", p_val) pollutant_str *= "CO, " end
        if occursin("NO", p_val) pollutant_str *= "NO2, " end
        if occursin("SO", p_val) pollutant_str *= "SO2, " end
        if occursin("10", p_val) pollutant_str *= "PM10, " end
        if occursin("2.5", p_val) pollutant_str *= "PM2.5, " end
        df[r, :pollutant] = rstrip(pollutant_str, [',', ' '])

        # Clean station count
        if !ismissing(df[r, :stations])
            s_val = replace(string(df[r, :stations]), " #" => "")
            df[r, :stations] = split(s_val, "/")[begin]
        end
    end

    # --- 4. Save the cleaned DataFrame back to the CSV file ---
    CSV.write(output_path, df)
    println("Successfully cleaned and saved data to: ", output_path)
end


"""
Main execution block
"""
function main()
    # Get today's date in Indian Standard Time (IST)
    ist_timezone = tz"Asia/Kolkata"
    today_date_ist = Date(now(ist_timezone))

    # Call the function to fetch and process the data for today
    fetch_and_process_bulletin(today_date_ist)
end

# Run the main function when the script is executed
main()
