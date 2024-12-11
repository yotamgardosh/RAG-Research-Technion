# -*- coding: utf-8 -*-

!pip install -U kaleido

import kaleido

from google.colab import drive
drive.mount('/content/drive')

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import drive
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

class LmHarnessVis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.drive_mounted = False
        if not self.drive_mounted:
            drive.mount('/content/drive')
            self.drive_mounted = True

    def mount_drive(self):
        if not self.drive_mounted:
            drive.mount('/content/drive')
            self.drive_mounted = True

    def read_res_json(self, file_name):
        json_data = []
        full_path = f"{self.file_path}/{file_name}"
        with open(full_path, 'r') as file:
            for line in file:
                json_data.append(json.loads(line))
        return json_data


    def create_res_df(self, json_data):
        data = []
        for entry in json_data:
            category = entry['doc']['Category']
            sub_category = entry['doc']['Sub-Category']
            responses = entry.get('filtered_resps', [])
            likelihoods = np.array([float(resp[0]) for resp in responses])
            true_indices = [i for i, resp in enumerate(responses) if resp[1] == "True"]

            # Apply softmax to the likelihoods to convert them to probabilities
            probabilities = self.softmax(likelihoods)
            confidence_percentages = probabilities * 100

            # Extract confidence percentages for 'True' answers
            true_confidences = [confidence_percentages[i] for i in true_indices]

            mean_confidence = np.mean(true_confidences) if true_confidences else 0
            correct = entry.get('acc', 0)

            data.append({
                'Category': category,
                'Sub-Category': sub_category,
                'Total Responses': len(responses),
                'True Responses': len(true_confidences),
                'Mean Confidence (%)': mean_confidence,
                'Correct': correct,

            })
        return pd.DataFrame(data)


    def calculate_accuracy(self, df, category_level='Sub-Category'):
        groupby_columns = ['Category', 'Sub-Category'] if category_level == 'Sub-Category' else ['Category']
        accuracy_data = df.groupby(groupby_columns).agg(
            Total_Questions=('Correct', 'size'),
            Correct_Answers=('Correct', 'sum')
        )
        accuracy_data['Accuracy (%)'] = (accuracy_data['Correct_Answers'] / accuracy_data['Total_Questions']) * 100
        return accuracy_data

    def total_accuracy(self, df):
        total_questions = df['Correct'].count()
        total_correct_answers = df['Correct'].sum()
        total_accuracy = (total_correct_answers / total_questions) * 100
        return total_accuracy

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def plot_average_likelihood(self, df, model_name, path=None):
        # Group by 'Sub-Category' to calculate mean and std of 'Mean Confidence (%)'
        stats = df.groupby('Sub-Category')['Mean Confidence (%)'].agg(['mean', 'std'])
        stats_sorted = stats.sort_values(by='mean', ascending=True)

        # Use a subtle color palette for each bar
        colors = [f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, 0.8)" for color in sns.color_palette("Paired", len(stats_sorted))]

        # Create a bar chart using plotly
        fig = go.Figure()

        # Add bars with mean shown above and std only in hover info
        fig.add_trace(go.Bar(
            x=stats_sorted.index,
            y=stats_sorted['mean'],
            marker=dict(color=colors),
            text=[f"{val:.2f}%" for val in stats_sorted['mean']],  # Mean to display above bars
            textposition='outside',  # Position mean text above bars
            hovertext=[f"Mean: {val:.2f}%<br>Std: {std:.2f}%" for val, std in zip(stats_sorted['mean'], stats_sorted['std'])],
            hoverinfo="text"  # Display mean and std in hover only
        ))

        # Update layout for readability
        fig.update_layout(
            title=f"{model_name} Confidence by Sub-Category",
            xaxis_title="Sub-Category",
            yaxis_title="Average Confidence (%) of Model Answers",
            yaxis=dict(range=[0, 110]),
            template="plotly_white",
            title_x=0.5,
            title_font_size=20
        )

                # Save the chart as SVG if a path is provided
        if path:
            # Use the title as the file name, replacing spaces with underscores
            file_name = f"{fig.layout.title.text.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as SVG
            fig.write_image(full_path, format="png", width=1600, height=1200, scale=2)

        fig.show()


        # Display the chart
        fig.show()



    def create_bar_chart(self, df, model_name, accuracy_df, path=None):
        # Ensure 'Category' is a column in accuracy_df
        if 'Category' not in accuracy_df.columns:
            accuracy_df.reset_index(inplace=True)

        # Define the custom color palette
        color_palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set2

        # Get unique categories
        categories = accuracy_df['Category'].unique()
        bar_width = 0.25
        fig = go.Figure()

        # Loop through each category and plot sub-category bars
        for i, category in enumerate(categories):
            sub_category_df = accuracy_df[accuracy_df['Category'] == category]

            # Assign colors from the custom palette
            color = color_palette[i % len(color_palette)]  # Cycle through palette if categories exceed colors

            # Positions and data for each category
            positions = np.arange(len(sub_category_df)) + i * bar_width
            fig.add_trace(go.Bar(
                x=sub_category_df['Sub-Category'],
                y=sub_category_df['Accuracy (%)'],
                name=category,
                text=[f"{acc:.1f}%" for acc in sub_category_df['Accuracy (%)']],
                textposition='outside',
                marker=dict(
                    color=color,  # Assign unique color
                    opacity=0.8,
                    line=dict(color='black', width=0.8)  # Add border to bars
                )
            ))

        # Update layout for readability
        fig.update_layout(
            barmode='group',  # Group bars by category
            title=f"{model_name} Accuracy Chart",
            xaxis_title="Sub-Categories",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 110]),
            legend_title_text='Categories',
            template="plotly_white",
            title_x=0.5,
            title_font_size=20
        )

        # Save the chart as PNG if a path is provided
        if path:
            # Use the title as the file name, replacing spaces with underscores
            file_name = f"{fig.layout.title.text.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as PNG
            fig.write_image(full_path, format="png", width=1200, height=900, scale=2)
        fig.show()





    def create_comparison_bar_chart(self, df_before, df_after, model_name_1, model_name_2, path=None):
        # Combine data to get categories and assign unique colors
        df_before['Model'] = model_name_1
        df_after['Model'] = model_name_2
        df_combined = pd.concat([df_before, df_after])

        # Get a list of unique sub-categories
        sub_categories = df_combined['Sub-Category'].unique()

        # Use a vibrant color palette
        color_palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set2
        num_colors_needed = len(sub_categories)
        colors = color_palette[:num_colors_needed]

        # Prepare figure
        fig = go.Figure()

        # Manually adjust x-axis positions for before and after bars
        x_positions_before = [i - 0.2 for i in range(len(sub_categories))]  # Slightly wider bars
        x_positions_after = [i + 0.2 for i in range(len(sub_categories))]

        # Iterate over each sub-category to add two bars with different transparency
        for i, sub_category in enumerate(sub_categories):
            sub_data_before = df_combined[(df_combined['Sub-Category'] == sub_category) & (df_combined['Model'] == model_name_1)]
            sub_data_after = df_combined[(df_combined['Sub-Category'] == sub_category) & (df_combined['Model'] == model_name_2)]

            if not sub_data_before.empty:
                fig.add_trace(go.Bar(
                    x=[x_positions_before[i]],
                    y=sub_data_before['Accuracy (%)'],
                    name=f"{model_name_1} (Before)",
                    marker_color=colors[i],
                    marker=dict(opacity=0.8),
                    width=0.4,  # Increased bar width for better visibility
                    text=[f"{acc:.1f}%" for acc in sub_data_before['Accuracy (%)']],
                    textposition='outside',
                    textfont=dict(size=16, color="black")
                ))

            if not sub_data_after.empty:
                fig.add_trace(go.Bar(
                    x=[x_positions_after[i]],
                    y=sub_data_after['Accuracy (%)'],
                    name=f"{model_name_2} (After)",
                    marker_color=colors[i],
                    marker=dict(opacity=0.5),
                    width=0.4,  # Increased bar width for better visibility
                    text=[f"{acc:.1f}%" for acc in sub_data_after['Accuracy (%)']],
                    textposition='outside',
                    textfont=dict(size=16, color="black")
                ))

        # Layout customization
        fig.update_layout(
            title=f"Comparison of {model_name_1} and {model_name_2} Accuracy by Sub-Category",
            xaxis=dict(
                title='Sub-Categories',
                tickvals=list(range(len(sub_categories))),
                ticktext=sub_categories,
                tickangle=45,  # Rotate labels diagonally in the opposite direction
                tickfont=dict(size=11)  # Smaller font size for sub-category labels
            ),
            yaxis=dict(
                title='Accuracy (%)',
                range=[0, 110],  # Extend range to go beyond 100%
                tickformat=".0f"
            ),
            showlegend=False,
            barmode='group',
            bargap=0,  # No gaps between grouped bars
            template="plotly_white",
            font=dict(size=14)  # General font size for axis labels
        )
        if path:
            # Use the title as the file name, replacing spaces with underscores
            file_name = f"{fig.layout.title.text.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as SVG
            fig.write_image(full_path, format="png", width=1200, height=900, scale=2)
        fig.show()


    def plot_compare_likelihood(self, df1, df2, model_name_1, model_name_2, path=None, with_std_bar=False):
        # Calculate mean and std for each model per sub-category
        stats1 = df1.groupby('Sub-Category')['Mean Confidence (%)'].agg(['mean', 'std'])
        stats2 = df2.groupby('Sub-Category')['Mean Confidence (%)'].agg(['mean', 'std'])

        # Join stats to combine both models' data in a single DataFrame
        combined_stats = stats1.join(stats2, lsuffix='_df1', rsuffix='_df2', how='outer').fillna(0)
        combined_stats_sorted = combined_stats.sort_values(by='mean_df1', ascending=True)

        # Create a plotly figure
        fig = go.Figure()

        # Add bars for the first model
        fig.add_trace(go.Bar(
            x=combined_stats_sorted.index,
            y=combined_stats_sorted['mean_df1'],
            name=model_name_1,
            error_y=dict(
                type='data',
                array=combined_stats_sorted['std_df1'] if with_std_bar else None,
                visible=with_std_bar  # Ensure this is a boolean
            ),
            marker_color='rgba(55, 128, 191, 0.7)',
            text=[f"{val:.1f}%" for val in combined_stats_sorted['mean_df1']],
            textposition='outside'
        ))

        # Add bars for the second model
        fig.add_trace(go.Bar(
            x=combined_stats_sorted.index,
            y=combined_stats_sorted['mean_df2'],
            name=model_name_2,
            error_y=dict(
                type='data',
                array=combined_stats_sorted['std_df2'] if with_std_bar else None,
                visible=with_std_bar  # Ensure this is a boolean
            ),
            marker_color='rgba(255, 153, 51, 0.7)',
            text=[f"{val:.1f}%" for val in combined_stats_sorted['mean_df2']],
            textposition='outside'
        ))

        # Update layout for grouped bars
        fig.update_layout(
            barmode='group',
            title=f"Compare {model_name_1} and {model_name_2} Confidence Score by Sub-Category",
            xaxis_title="Sub-Category",
            yaxis_title="Mean Confidence (%) in Model Answer",
            yaxis=dict(range=[0, 110]),
            template="plotly_white",
            title_x=0.5,
            title_font_size=20
        )

        # Save the chart as SVG if a path is provided
        if path:
            # Use the title text as the file name
            file_name = f"{fig.layout.title.text.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as SVG
            fig.write_image(full_path, format="png", width=1200, height=900, scale=2)

        # Show the figure
        fig.show()



    def create_comparison_table(self, df1, df2, df3, df4, model_name_1, model_name_2, model_name_3, model_name_4, title, path=None):
        # Group each DataFrame by 'Sub-Category' and count occurrences
        stats1 = df1.groupby('Sub-Category').size().reset_index(name='Question Count')
        stats1['Question %'] = (stats1['Question Count'] / stats1['Question Count'].sum() * 100).map("{:.2f}%".format)

        # Summarize the number of correct answers per model
        model1_results = df1.groupby('Sub-Category')['Correct'].sum().rename(model_name_1)
        model2_results = df2.groupby('Sub-Category')['Correct'].sum().rename(model_name_2)
        model3_results = df3.groupby('Sub-Category')['Correct'].sum().rename(model_name_3)
        model4_results = df4.groupby('Sub-Category')['Correct'].sum().rename(model_name_4)

        # Combine all results into a single DataFrame
        comparison_df = stats1.set_index('Sub-Category').join([model1_results, model2_results, model3_results, model4_results])

        # Calculate percentage of correct answers for each model in each sub-category
        comparison_df[f'{model_name_1} %'] = (comparison_df[model_name_1] / comparison_df['Question Count'] * 100).fillna(0).map("{:.2f}%".format)
        comparison_df[f'{model_name_2} %'] = (comparison_df[model_name_2] / comparison_df['Question Count'] * 100).fillna(0).map("{:.2f}%".format)
        comparison_df[f'{model_name_3} %'] = (comparison_df[model_name_3] / comparison_df['Question Count'] * 100).fillna(0).map("{:.2f}%".format)
        comparison_df[f'{model_name_4} %'] = (comparison_df[model_name_4] / comparison_df['Question Count'] * 100).fillna(0).map("{:.2f}%".format)

        # Add a total row at the bottom
        total_row = pd.DataFrame({
            'Question Count': [comparison_df['Question Count'].sum()],
            'Question %': ['100.00%'],
            model_name_1: [comparison_df[model_name_1].sum()],
            model_name_2: [comparison_df[model_name_2].sum()],
            model_name_3: [comparison_df[model_name_3].sum()],
            model_name_4: [comparison_df[model_name_4].sum()],
            f'{model_name_1} %': [f"{(comparison_df[model_name_1].sum() / comparison_df['Question Count'].sum() * 100):.2f}%"],
            f'{model_name_2} %': [f"{(comparison_df[model_name_2].sum() / comparison_df['Question Count'].sum() * 100):.2f}%"],
            f'{model_name_3} %': [f"{(comparison_df[model_name_3].sum() / comparison_df['Question Count'].sum() * 100):.2f}%"],
            f'{model_name_4} %': [f"{(comparison_df[model_name_4].sum() / comparison_df['Question Count'].sum() * 100):.2f}%"]
        }, index=['Grand Total'])

        # Append the total row
        comparison_df = pd.concat([comparison_df, total_row])

        # Reset index for display
        comparison_df.reset_index(inplace=True)

        # Create the Plotly table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(comparison_df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)  # Larger font size for headers
            ),
            cells=dict(
                values=[comparison_df[col] for col in comparison_df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=12)  # Larger font size for cells
            )
        )])

        fig.update_layout(title_text=title, title_x=0.5, title_font=dict(size=18))  # Larger title font size

        if path:
            # Use the title as the file name, replacing spaces with underscores
            file_name = f"{title.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as PNG
            fig.write_image(full_path, format="png", width=1200, height=900, scale=2)

        fig.show()
        return fig





    def plot_confidence_accuracy_correlation(self, df, title,path=None):
        # Calculate mean confidence and accuracy per sub-category
        confidence_data = df.groupby('Sub-Category')['Mean Confidence (%)'].mean()
        accuracy_data = df.groupby('Sub-Category')['Correct'].mean() * 100  # Convert accuracy to percentage

        # Combine into a DataFrame
        correlation_df = pd.DataFrame({
            'Sub-Category': confidence_data.index,
            'Mean Confidence (%)': confidence_data.values,
            'Accuracy (%)': accuracy_data.values
        })

        # Calculate correlation
        correlation = correlation_df['Mean Confidence (%)'].corr(correlation_df['Accuracy (%)'])
        correlation_text = f"Correlation: {correlation:.2f}"

        # Create an interactive scatter plot with sub-category color and a distinct trend line
        fig = px.scatter(
            correlation_df,
            x='Mean Confidence (%)',
            y='Accuracy (%)',
            color='Sub-Category',  # Different color for each sub-category
            title=f"{title} <br>{correlation_text}",
            labels={"Mean Confidence (%)": "Mean Confidence (%)", "Accuracy (%)": "Accuracy (%)"}
        )

        # Add a trendline using OLS and customize its color
        trendline = px.scatter(correlation_df, x='Mean Confidence (%)', y='Accuracy (%)', trendline="ols")
        trendline.data[1].marker.color = 'black'  # Change trend line color
        trendline.data[1].name = 'Trend Line'     # Label for trend line
        fig.add_trace(trendline.data[1])  # Add trend line to the main plot

        # Customize layout for readability
        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            template="plotly_white",
            showlegend=True
        )
        if path:
            # Use the title as the file name, replacing spaces with underscores
            file_name = f"{title.replace(' ', '_')}.png"
            full_path = f"{path}{file_name}"

            # Save the plot as SVG
            fig.write_image(full_path, format="png", width=1200, height=900, scale=2)
        fig.show()
