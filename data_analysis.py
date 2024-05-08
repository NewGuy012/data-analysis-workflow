import os.path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Check if cleansed file already exists
path = "./ev_dataset_washington_cleansed.parquet"

if not os.path.isfile(path):
    # Import csv as dataframe
    df = pd.read_csv("ev_dataset_washington.csv").convert_dtypes()

    # Update column names
    new_column_names = {
        "VIN (1-10)": "vin",
        "County": "county",
        "City": "city",
        "State": "state",
        "Postal Code": "postal_code",
        "Model Year": "year",
        "Make": "make",
        "Model": "model",
        "Electric Vehicle Type": "vehicle_type",
        "Electric Range": "range",
        "Base MSRP": "base_msrp",
        "DOL Vehicle ID": "vehicle_id",
        "Electric Utility": "electric_utility",
    }
    df = df.rename(columns=new_column_names)

    # Remove rows with NaNs
    remove_index = df.loc[df.isna().any(axis="columns")].index
    df = df.drop(remove_index)

    # Clean up certain columns
    df = df.assign(
        electric_utility=lambda df: (
            df["electric_utility"].str.removesuffix(" - (WA)")
        ),
        model=lambda df: (df["model"].str.capitalize()),
    )
    df = df.drop_duplicates(ignore_index=True)

    # Verify data
    df.info()
    df["make"].value_counts()
    df["base_msrp"].describe()
    df.loc[df.duplicated(keep=False)]

    # Export to parquet format
    df.to_parquet("ev_dataset_washington_cleansed.parquet", index=False)
else:
    # Import csv as dataframe
    df = pd.read_parquet("ev_dataset_washington_cleansed.parquet")

# Plot
fig, ax = plt.subplots()
ax.scatter(df["make"], df["range"])
ax.set_title("Make vs Electric Range")
ax.set_xlabel("Make")
ax.set_ylabel("Electric Range")
ax.tick_params(axis="x", labelrotation=90)
# fig.show()

fig, ax = plt.subplots()
ax.hist(df["make"])
ax.set_title("Histogram of Make")
ax.set_xlabel("Make")
ax.set_ylabel("Count")
ax.tick_params(axis="x", labelrotation=90)
# fig.show()

fig, ax = plt.subplots()
ax.hist(df["range"])
ax.set_title("Histogram of Electric Range")
ax.set_xlabel("Electric Range")
ax.set_ylabel("Count")
ax.tick_params(axis="x", labelrotation=90)
# fig.show()

fig, ax = plt.subplots()
df_pie = df["vehicle_type"].value_counts()
labels = df_pie.index
values = df_pie.values
ax.pie(values, labels=labels, autopct="%1.1f%%")
ax.set_title("Pie Chart of Vehicle Type")

# Determine common statistics
df["range"].agg(["min", "max", "mean", "std"])

# Regression analysis
df_regress = df[["base_msrp", "range", "vehicle_type"]]

# Remove rows
remove_index = df_regress.loc[
    (df_regress.vehicle_type == "Plug-in Hybrid Electric Vehicle (PHEV)")
].index
df_regress = df_regress.drop(remove_index)

remove_index = df_regress.loc[(df_regress.range == 0)].index
df_regress = df_regress.drop(remove_index)

remove_index = df_regress.loc[
    (df_regress.base_msrp == df_regress.base_msrp.max()) | (df_regress.base_msrp == 0)
].index
df_regress = df_regress.drop(remove_index)

x = df_regress.loc[:, ["base_msrp"]]
y = df_regress.loc[:, "range"]

model = LinearRegression()
model.fit(x, y)

r_squared = f"R-Squared: {model.score(x, y):.2f}"
best_fit = f"y = {model.coef_[0]:.4f}x{model.intercept_:+.4f}"
y_pred = model.predict(x)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, y_pred, color="red")
# ax.text(7.25, 5.5, r_squared, fontsize=10)
# ax.text(7.25, 7, best_fit, fontsize=10)
ax.set_title("MSRP vs Electric Range")
ax.set_xlabel("Base MSRP")
ax.set_ylabel("Electric Range")
# fig.show()

plt.show(block=True)
