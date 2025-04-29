"""
Chart renderer for generating chart images from JSON specifications.
"""
import os
import json
import uuid
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any, Optional, List, Union, Tuple

# Create visualization directory if it doesn't exist
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def hex_to_rgba(color_str: str, alpha: float = 1.0) -> Union[Tuple[float, float, float, float], str]:
    """Convert color string to RGBA tuple."""
    try:
        # Check if it's already an rgba string
        if color_str.startswith('rgba('):
            # Extract the rgba values using regex
            rgba_match = re.match(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)', color_str)
            if rgba_match:
                r, g, b, a = rgba_match.groups()
                return (int(r)/255, int(g)/255, int(b)/255, float(a))
            else:
                return mcolors.to_rgba("#36a2eb", alpha)
        # Check if it's an rgb string
        elif color_str.startswith('rgb('):
            # Extract the rgb values using regex
            rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
            if rgb_match:
                r, g, b = rgb_match.groups()
                return (int(r)/255, int(g)/255, int(b)/255, alpha)
            else:
                return mcolors.to_rgba("#36a2eb", alpha)
        # Otherwise, assume it's a hex color or named color
        else:
            return mcolors.to_rgba(color_str, alpha)
    except ValueError:
        # Return a default color if conversion fails
        print(f"Warning: Could not convert color '{color_str}' to RGBA, using default color")
        return mcolors.to_rgba("#36a2eb", alpha)

def render_chart(chart_json: str) -> Optional[str]:
    """
    Render a chart from a JSON specification and save it to the file system.
    Returns the path to the saved image file, or None if rendering fails.
    """
    try:
        # Parse the chart JSON
        if isinstance(chart_json, str):
            chart_spec = json.loads(chart_json)
        else:
            chart_spec = chart_json
            
        # Extract chart data
        chart_type = chart_spec.get("type", "bar")
        
        # Handle different JSON structures
        if "data" in chart_spec and "labels" in chart_spec["data"]:
            # Chart.js style structure
            labels = chart_spec["data"]["labels"]
            datasets = chart_spec["data"]["datasets"]
            title = chart_spec.get("options", {}).get("title", {}).get("text", "Chart")
            if isinstance(title, dict) and "text" in title:
                title = title["text"]
        else:
            # Our custom structure
            labels = chart_spec.get("labels", [])
            datasets = chart_spec.get("datasets", [])
            title = chart_spec.get("title", "Chart")
        
        # Create figure and axis
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Render different chart types
        if chart_type.lower() == "pie":
            render_pie_chart(ax, labels, datasets, title)
        elif chart_type.lower() == "line":
            render_line_chart(ax, labels, datasets, title)
        else:  # Default to bar chart
            render_bar_chart(ax, labels, datasets, title)
        
        # Generate a unique filename
        filename = f"chart_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(VISUALIZATION_DIR, filename)
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    except Exception as e:
        print(f"Error rendering chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def render_bar_chart(ax, labels: List[str], datasets: List[Dict[str, Any]], title: str):
    """Render a bar chart."""
    num_datasets = len(datasets)
    num_labels = len(labels)
    bar_width = 0.8 / num_datasets if num_datasets > 0 else 0.8
    
    for i, dataset in enumerate(datasets):
        data = dataset.get("data", [])
        label = dataset.get("label", f"Dataset {i+1}")
        
        # Get colors
        if "backgroundColor" in dataset:
            if isinstance(dataset["backgroundColor"], list):
                colors = [hex_to_rgba(color) for color in dataset["backgroundColor"]]
                # If we don't have enough colors, cycle through them
                colors = colors * (len(data) // len(colors) + 1)
                colors = colors[:len(data)]
            else:
                colors = hex_to_rgba(dataset["backgroundColor"])
        else:
            colors = [plt.cm.tab10(i / 10) for _ in range(len(data))]
        
        # Calculate x positions for bars
        x = list(range(len(data)))
        if num_datasets > 1:
            x = [pos + (i - num_datasets/2 + 0.5) * bar_width for pos in x]
        
        # Plot bars
        ax.bar(x, data, width=bar_width, label=label, color=colors)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xticks(range(num_labels))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Add legend if multiple datasets
    if num_datasets > 1:
        ax.legend()

def render_pie_chart(ax, labels: List[str], datasets: List[Dict[str, Any]], title: str):
    """Render a pie chart."""
    if not datasets:
        return
    
    # Use the first dataset for pie chart
    dataset = datasets[0]
    data = dataset.get("data", [])
    
    # Get colors
    if "backgroundColor" in dataset and isinstance(dataset["backgroundColor"], list):
        colors = [hex_to_rgba(color) for color in dataset["backgroundColor"]]
    else:
        colors = plt.cm.tab10.colors
    
    # Plot pie chart
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title)

def render_line_chart(ax, labels: List[str], datasets: List[Dict[str, Any]], title: str):
    """Render a line chart."""
    for i, dataset in enumerate(datasets):
        data = dataset.get("data", [])
        label = dataset.get("label", f"Dataset {i+1}")
        
        # Get line color
        if "borderColor" in dataset:
            if isinstance(dataset["borderColor"], list):
                color = hex_to_rgba(dataset["borderColor"][0])
            else:
                color = hex_to_rgba(dataset["borderColor"])
        else:
            color = plt.cm.tab10(i / 10)
        
        # Plot line
        ax.plot(range(len(data)), data, label=label, color=color, marker='o')
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Add legend if multiple datasets
    if len(datasets) > 1:
        ax.legend()
