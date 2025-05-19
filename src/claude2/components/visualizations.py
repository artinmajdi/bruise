import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_timeline(events, title="Project Timeline"):
    """
    Creates a horizontal timeline visualization from a list of events.
    
    Args:
        events: List of dicts with 'event', 'start', and 'duration' keys
        title: Title for the timeline chart
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Convert events to DataFrame
    df = pd.DataFrame(events)
    
    # Plot timeline
    ax.barh(df['event'], df['duration'], left=df['start'], height=0.5)
    
    # Add labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['start'] + row['duration']/2, i, row['event'], 
                ha='center', va='center')
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.get_yaxis().set_visible(False)
    
    return fig

def create_skill_radar(skills, values, title="Technical Skills"):
    """
    Creates a radar chart for skill visualization.
    
    Args:
        skills: List of skill names
        values: List of skill values (same length as skills)
        title: Title for the radar chart
    
    Returns:
        Matplotlib figure object
    """
    categories = skills
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create values for radar chart
    values = values + values[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.1)
    
    # Add labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    ax.set_title(title)
    ax.grid(True)
    
    return fig

def create_team_card(name, role, expertise, image=None):
    """
    Creates a card UI for team member information.
    
    Args:
        name: Team member name
        role: Team member role
        expertise: List of expertise areas
        image: Optional image path
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if image:
            st.image(image, width=150)
        else:
            st.markdown("👤")
    
    with col2:
        st.subheader(name)
        st.write(f"**Role:** {role}")
        st.write("**Expertise:**")
        for item in expertise:
            st.write(f"- {item}")
