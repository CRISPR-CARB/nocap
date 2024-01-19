import marimo

__generated_with = "0.1.78"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    print("hello world")
    return (mo,)


@app.cell
def __(mo):
    X = mo.ui.slider(10, 1000, step=1, label=r"$\chi$")
    X
    return (X,)


@app.cell
def __(X):
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.random.normal(0, 1, size=int(X.value))
    plt.hist(y, bins=10)
    plt.show()
    return np, plt, y


@app.cell
def __(mo):
    import altair as alt
    import vega_datasets

    # Load some data
    cars = vega_datasets.data.cars()

    # Create an Altair chart
    chart = (
        alt.Chart(cars)
        .mark_point()
        .encode(
            x="Horsepower",  # Encoding along the x-axis
            y="Miles_per_Gallon",  # Encoding along the y-axis
            color="Origin",  # Category encoding by color
        )
    )

    # Make it reactive ⚡
    chart = mo.ui.altair_chart(chart)
    return alt, cars, chart, vega_datasets


@app.cell
def __(chart, mo):
    mo.vstack([chart, chart.value.head()])
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
