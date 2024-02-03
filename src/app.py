import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")


class CSVPlotter:

    def __init__(self):
        self.df_list = []
        self.file_names = []
        self.run()

    def plot_csv_data(self, df, file_name, log_view):
        if log_view:
            # Logarithmic view
            fig = px.line(df, x=df.columns[0], y=df.columns[1:], log_y=True)
        else:
            # Normal view
            fig = px.line(df, x=df.columns[0], y=df.columns[1:])

        # グラフの下限を0に設定
        fig.update_layout(title=file_name, legend_title_text='Labels')
        return fig

    def run(self):
        st.title('CSV Multi-File Plotter')

        uploaded_files = st.sidebar.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

        num_graphs_horizontal = st.sidebar.number_input('Number of graphs in a row', min_value=1, value=1)

        # Add a toggle in the sidebar to select the view mode
        log_view = st.sidebar.checkbox('Logarithmic View', value=True)  # This checkbox returns True if selected

        if uploaded_files:
            for uploaded_file in uploaded_files:
                df = pd.read_csv(uploaded_file)
                self.df_list.append(df)
                self.file_names.append(uploaded_file.name)

        total_files = len(self.df_list)

        if st.sidebar.button('Plot Data') and total_files > 0:
            rows = (total_files + num_graphs_horizontal - 1) // num_graphs_horizontal
            for r in range(rows):
                cols = st.columns(num_graphs_horizontal)
                for c in range(num_graphs_horizontal):
                    idx = r * num_graphs_horizontal + c
                    if idx < total_files:
                        cols[c].plotly_chart(self.plot_csv_data(self.df_list[idx], self.file_names[idx], log_view),
                                             use_container_width=True)


if __name__ == '__main__':
    CSVPlotter()
