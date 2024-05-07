import pandas as pd

james_bond_data = pd.read_csv("james_bond_data.csv").convert_dtypes()

new_column_names = {
    "Release": "release_date",
    "Movie": "movie_title",
    "Bond": "bond_actor",
    "Bond_Car_MFG": "car_manufacturer",
    "US_Gross": "income_usa",
    "World_Gross": "income_world",
    "Budget ($ 000s)": "movie_budget",
    "Film_Length": "film_length",
    "Avg_User_IMDB": "imdb",
    "Avg_User_Rtn_Tom": "rotten_tomatoes",
    "Martinis": "martinis_consumed",
    "Kills_Bond": "bond_kills",
}

data = james_bond_data.rename(columns=new_column_names)

data = (
    james_bond_data.rename(columns=new_column_names)
    .combine_first(
        pd.DataFrame({"imdb": {10: 7.1}, "rotten_tomatoes": {10: 6.8}})
    )
    .assign(
        income_usa=lambda data: (
            data["income_usa"]
            .replace("[$,]", "", regex=True)
            .astype("Float64")
        ),
        income_world=lambda data: (
            data["income_world"]
            .replace("[$,]", "", regex=True)
            .astype("Float64")
        ),
        movie_budget=lambda data: (
            data["movie_budget"]
            .replace("[$,]", "", regex=True)
            .astype("Float64")
            * 1000
        ),
        film_length=lambda data: (
            data["film_length"]
            .str.removesuffix("mins")
            .astype("Int64")
            .replace(1200, 120)
        ),
        release_date=lambda data: pd.to_datetime(
            data["release_date"], format="%B, %Y"
        ),
        release_year=lambda data: data["release_date"].dt.year.astype("Int64"),
        bond_actor=lambda data: (
            data["bond_actor"]
            .str.replace("Shawn", "Sean")
            .str.replace("MOORE", "Moore")
        ),
        car_manufacturer=lambda data: data["car_manufacturer"].str.replace(
            "Astin", "Aston"
        ),
        martinis_consumed=lambda data: data["martinis_consumed"].replace(
            -6, 6
        ),
    )
    .drop_duplicates(ignore_index=True)
)

data.to_csv("james_bond_data_cleansed.csv", index=False)

# Performing a Regression Analysis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = data.loc[:, ["imdb"]]
y = data.loc[:, "rotten_tomatoes"]

model = LinearRegression()
model.fit(x, y)

r_squared = f"R-Squared: {model.score(x, y):.2f}"
best_fit = f"y = {model.coef_[0]:.4f}x{model.intercept_:+.4f}"
y_pred = model.predict(x)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, y_pred, color="red")
ax.text(7.25, 5.5, r_squared, fontsize=10)
ax.text(7.25, 7, best_fit, fontsize=10)
ax.set_title("Scatter Plot of Ratings")
ax.set_xlabel("Average IMDb Rating")
ax.set_ylabel("Average Rotten Tomatoes Rating")
# fig.show()

# Investigating a Statistical Distribution
fig, ax = plt.subplots()
length = data["film_length"].value_counts(bins=7).sort_index()
length.plot.bar(
    ax=ax,
    title="Film Length Distribution",
    xlabel="Time Range (mins)",
    ylabel="Count",
)
# fig.show()

data["film_length"].agg(["min", "max", "mean", "std"])

# Finding No Relationship
fig, ax = plt.subplots()
ax.scatter(data["imdb"], data["bond_kills"])
ax.set_title("Scatter Plot of Kills vs Ratings")
ax.set_xlabel("Average IMDb Rating")
ax.set_ylabel("Kills by Bond")
# fig.show()