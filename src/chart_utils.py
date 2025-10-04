import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


def preprocess_dataframe(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Preprocess DataFrame with date conversion, sorting, and basic cleaning."""
    if df.empty:
        return df

    # Convert date column to datetime if it exists
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

    return df


def get_trend_color(
    trend: str, confidence: float, high_confidence_threshold: float = 0.7
) -> str:
    """Get color based on trend and confidence level."""
    if trend == "UP":
        return "#2ca02c" if confidence > high_confidence_threshold else "#98df8a"
    elif trend == "DOWN":
        return "#d62728" if confidence > high_confidence_threshold else "#ff9999"
    else:
        return "#ff7f0e"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to RGBA format with transparency."""
    # Remove # if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f"rgba({r}, {g}, {b}, {alpha})"


def create_animation_controls(
    duration: int = 500,
    transition_duration: int = 300,
    button_prefix: str = "Frame: ",
    slider_prefix: str = "Day ",
) -> Dict[str, Any]:
    """Create standardized animation controls for Plotly charts."""
    return {
        "updatemenus": [
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                None,
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": transition_duration},
                                },
                            ],
                            label="▶️ Play",
                            method="animate",
                        ),
                        dict(
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            label="⏸️ Pause",
                            method="animate",
                        ),
                    ]
                ),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
        "sliders": [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": button_prefix},
                transition={"duration": transition_duration},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[],  # Will be populated with frame-specific steps
            )
        ],
    }


def create_chart_layout(
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 500,
    showlegend: bool = True,
    hovermode: str = "x unified",
) -> Dict[str, Any]:
    """Create common chart layout configuration."""
    return {
        "title": title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "hovermode": hovermode,
        "showlegend": showlegend,
        "height": height,
    }


def create_price_traces(
    price_df: pd.DataFrame, ind_df: Optional[pd.DataFrame] = None
) -> List[go.Scatter]:
    """Create price and indicator traces for charts."""
    traces = []

    # Add price line
    if not price_df.empty:
        traces.append(
            go.Scatter(
                x=price_df["date"],
                y=price_df["close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#1f77b4", width=3),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            )
        )

    # Add moving averages
    if ind_df is not None and not ind_df.empty:
        traces.append(
            go.Scatter(
                x=ind_df["date"],
                y=ind_df["ma5"],
                mode="lines",
                name="MA5",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="Date: %{x}<br>MA5: $%{y:.2f}<extra></extra>",
            )
        )

        traces.append(
            go.Scatter(
                x=ind_df["date"],
                y=ind_df["ma10"],
                mode="lines",
                name="MA10",
                line=dict(color="#2ca02c", width=2, dash="dot"),
                hovertemplate="Date: %{x}<br>MA10: $%{y:.2f}<extra></extra>",
            )
        )

    return traces


def create_forecast_traces(
    forecast_df: pd.DataFrame, line_color: str
) -> List[go.Scatter]:
    """Create forecast traces with confidence band."""
    traces = []

    # Add main forecast line
    traces.append(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_price"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=line_color, width=3),
            marker=dict(size=8, color=line_color, symbol="circle"),
            hovertemplate="Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>",
        )
    )

    # Add confidence band if we have more than 1 point
    if len(forecast_df) > 1:
        confidence_values = [
            0.5 if pd.isna(x) or x is None else x for x in forecast_df["confidence"]
        ]
        forecast_prices = [
            0.0 if pd.isna(x) or x is None else x for x in forecast_df["forecast_price"]
        ]

        upper_bound = [
            fp * (1 + (1 - cv) * 0.1) if fp is not None and cv is not None else 0
            for fp, cv in zip(forecast_prices, confidence_values)
        ]
        lower_bound = [
            fp * (1 - (1 - cv) * 0.1) if fp is not None and cv is not None else 0
            for fp, cv in zip(forecast_prices, confidence_values)
        ]

        fill_color = hex_to_rgba(line_color, 0.2)

        traces.append(
            go.Scatter(
                x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
                y=upper_bound + lower_bound[::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Band",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def create_animation_frames(
    df: pd.DataFrame,
    trace_creator_func,
    ind_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> List[go.Frame]:
    """Create animation frames for a given DataFrame and trace creation function."""
    frames = []

    for i in range(len(df)):
        frame_data = []
        current_df = df.iloc[: i + 1]

        if len(current_df) > 0:
            if ind_df is not None:
                # Use synchronized indicator data
                ind_points = min(i + 1, len(ind_df))
                current_ind_df = ind_df.iloc[:ind_points] if ind_points > 0 else None
                frame_data = trace_creator_func(current_df, current_ind_df, **kwargs)
            else:
                frame_data = trace_creator_func(current_df, **kwargs)

        frames.append(go.Frame(data=frame_data, name=str(i)))

    return frames
