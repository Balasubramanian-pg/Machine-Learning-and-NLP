"""
Character Network Analysis
-------------------------
This module provides tools for analyzing character relationships in text.
It extracts named entities, identifies character interactions, and creates
network visualizations to analyze character importance and communities.
"""

import os
import re
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import community as community_louvain


class CharacterNetworkAnalyzer:
    """Analyzes character networks in text using NLP and graph theory."""
    
    def __init__(self, language_model="en_core_web_sm"):
        """
        Initialize the analyzer with a spaCy language model.
        
        Args:
            language_model (str): The spaCy language model to use
        """
        # Load spaCy language model
        try:
            self.nlp = spacy.load(language_model)
        except OSError:
            print(f"Downloading language model: {language_model}")
            os.system(f"python -m spacy download {language_model}")
            self.nlp = spacy.load(language_model)
            
        self.characters_df = None
        self.book_docs = {}
        self.graphs = {}
        
    def load_characters(self, character_file):
        """
        Load character names from a CSV file.
        
        Args:
            character_file (str): Path to the CSV file containing character names
            
        Returns:
            pd.DataFrame: DataFrame containing character information
        """
        self.characters_df = pd.read_csv(character_file)
        
        # Clean character names
        self.characters_df['character'] = self.characters_df['character'].apply(
            lambda x: re.sub(r"[\(].*?[\)]", "", x).strip()
        )
        self.characters_df['character_firstname'] = self.characters_df['character'].apply(
            lambda x: x.split(' ', 1)[0]
        )
        
        return self.characters_df
    
    def load_books(self, data_dir):
        """
        Load all book files from a directory.
        
        Args:
            data_dir (str): Directory containing book text files
            
        Returns:
            list: List of book file paths
        """
        self.book_files = [b for b in os.scandir(data_dir) if '.txt' in b.name]
        self.book_files.sort(key=lambda x: x.name)  # Sort books by name
        return self.book_files
    
    def process_book(self, book_file):
        """
        Process a book file and extract named entities.
        
        Args:
            book_file: Path to book file
            
        Returns:
            spacy.Doc: Processed document with named entities
        """
        book_name = os.path.basename(book_file.name)
        print(f"Processing {book_name}...")
        
        book_text = open(book_file).read()
        book_doc = self.nlp(book_text)
        self.book_docs[book_name] = book_doc
        
        return book_doc
    
    def visualize_entities(self, doc, start=0, end=2000):
        """
        Visualize named entities in a document.
        
        Args:
            doc (spacy.Doc): Document to visualize
            start (int): Start index
            end (int): End index
        """
        return displacy.render(doc[start:end], style="ent", jupyter=True)
    
    def extract_entities_per_sentence(self, doc):
        """
        Extract named entities for each sentence in a document.
        
        Args:
            doc (spacy.Doc): Document to process
            
        Returns:
            pd.DataFrame: DataFrame with sentences and their entities
        """
        sent_entity_data = []
        
        # Loop through sentences, store named entity list for each sentence
        for sent in doc.sents:
            entity_list = [ent.text for ent in sent.ents]
            sent_entity_data.append({"sentence": sent, "entities": entity_list})
            
        return pd.DataFrame(sent_entity_data)
    
    def filter_character_entities(self, sentence_entity_df):
        """
        Filter named entities to only include characters.
        
        Args:
            sentence_entity_df (pd.DataFrame): DataFrame with sentences and entities
            
        Returns:
            pd.DataFrame: DataFrame with filtered character entities
        """
        if self.characters_df is None:
            raise ValueError("Characters must be loaded first using load_characters()")
        
        # Function to filter entities to only include characters
        def filter_entity(entity_list, character_df):
            return [entity for entity in entity_list 
                    if any(character in entity for character in character_df['character'].values)]
        
        # Apply filtering to all sentences
        sentence_entity_df['character_entities'] = sentence_entity_df['entities'].apply(
            lambda x: filter_entity(x, self.characters_df)
        )
        
        # Filter out sentences that don't have any character entities
        filtered_df = sentence_entity_df[sentence_entity_df['character_entities'].map(len) > 0].copy()
        
        # Take only first name of characters
        filtered_df['character_entities'] = filtered_df['character_entities'].apply(
            lambda x: [item.split()[0] for item in x]
        )
        
        return filtered_df
    
    def create_relationships(self, filtered_df, window_size=5):
        """
        Create character relationships based on co-occurrence within a window.
        
        Args:
            filtered_df (pd.DataFrame): DataFrame with filtered character entities
            window_size (int): Window size for co-occurrence
            
        Returns:
            pd.DataFrame: DataFrame with source-target relationships
        """
        relationships = []
        
        for i in range(filtered_df.index[-1]):
            end_i = min(i + window_size, filtered_df.index[-1])
            char_list = sum((filtered_df.loc[i:end_i].character_entities), [])
            
            # Remove duplicated characters that are next to each other
            char_unique = [char_list[i] for i in range(len(char_list)) 
                       if (i == 0) or char_list[i] != char_list[i-1]]
            
            if len(char_unique) > 1:
                for idx, source in enumerate(char_unique[:-1]):
                    target = char_unique[idx + 1]
                    relationships.append({"source": source, "target": target, "value": 1})
        
        # Create DataFrame from relationships
        relationship_df = pd.DataFrame(relationships)
        
        if not relationship_df.empty:
            # Sort the source and target to avoid duplicates like a->b and b->a
            relationship_df[['source', 'target']] = np.sort(
                relationship_df[['source', 'target']].values, axis=1
            )
            
            # Aggregate duplicate relationships
            relationship_df = relationship_df.groupby(['source', 'target'])['value'].sum().reset_index()
        
        return relationship_df
    
    def create_graph(self, relationship_df):
        """
        Create a network graph from relationship DataFrame.
        
        Args:
            relationship_df (pd.DataFrame): DataFrame with character relationships
            
        Returns:
            networkx.Graph: Character network graph
        """
        G = nx.from_pandas_edgelist(
            relationship_df,
            source="source",
            target="target",
            edge_attr="value",
            create_using=nx.Graph()
        )
        
        return G
    
    def analyze_book(self, book_file):
        """
        Perform full analysis on a book.
        
        Args:
            book_file: Path to book file
            
        Returns:
            tuple: (filtered_df, relationship_df, graph)
        """
        book_name = os.path.basename(book_file.name)
        
        # Process book
        book_doc = self.process_book(book_file)
        
        # Extract entities per sentence
        sent_entity_df = self.extract_entities_per_sentence(book_doc)
        
        # Filter for character entities
        filtered_df = self.filter_character_entities(sent_entity_df)
        
        # Create character relationships
        relationship_df = self.create_relationships(filtered_df)
        
        # Create graph
        graph = self.create_graph(relationship_df)
        self.graphs[book_name] = graph
        
        return filtered_df, relationship_df, graph
    
    def analyze_all_books(self):
        """
        Analyze all loaded books.
        
        Returns:
            dict: Dictionary of book graphs
        """
        if not hasattr(self, 'book_files'):
            raise ValueError("Books must be loaded first using load_books()")
        
        for book_file in self.book_files:
            self.analyze_book(book_file)
            
        return self.graphs
    
    def visualize_graph(self, graph, figsize=(10, 10)):
        """
        Visualize a character network graph using matplotlib.
        
        Args:
            graph (networkx.Graph): Graph to visualize
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, with_labels=True, node_color='skyblue', 
                edge_cmap=plt.cm.Blues, pos=pos)
        plt.show()
    
    def visualize_interactive_graph(self, graph, output_file="character_network.html", 
                                   width="1000px", height="700px", 
                                   bgcolor='#222222', font_color='white'):
        """
        Create an interactive visualization of the character network.
        
        Args:
            graph (networkx.Graph): Graph to visualize
            output_file (str): Output HTML file name
            width (str): Width of the visualization
            height (str): Height of the visualization
            bgcolor (str): Background color
            font_color (str): Font color
            
        Returns:
            pyvis.network.Network: Interactive network object
        """
        net = Network(notebook=True, width=width, height=height, 
                     bgcolor=bgcolor, font_color=font_color)
        
        # Set node size based on degree
        node_degree = dict(graph.degree)
        nx.set_node_attributes(graph, node_degree, 'size')
        
        # Add the graph to the visualization
        net.from_nx(graph)
        net.show(output_file)
        
        return net
    
    def calculate_centrality_measures(self, graph):
        """
        Calculate various centrality measures for a graph.
        
        Args:
            graph (networkx.Graph): Graph to analyze
            
        Returns:
            dict: Dictionary of DataFrames with centrality measures
        """
        # Degree centrality
        degree_dict = nx.degree_centrality(graph)
        degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
        
        # Betweenness centrality
        betweenness_dict = nx.betweenness_centrality(graph)
        betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
        
        # Closeness centrality
        closeness_dict = nx.closeness_centrality(graph)
        closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
        
        # Add centrality measures to the graph
        nx.set_node_attributes(graph, degree_dict, 'degree_centrality')
        nx.set_node_attributes(graph, betweenness_dict, 'betweenness_centrality')
        nx.set_node_attributes(graph, closeness_dict, 'closeness_centrality')
        
        return {
            'degree': degree_df,
            'betweenness': betweenness_df,
            'closeness': closeness_df
        }
    
    def plot_top_characters(self, centrality_df, title="Top Characters", n=10):
        """
        Plot top characters by centrality measure.
        
        Args:
            centrality_df (pd.DataFrame): DataFrame with centrality measures
            title (str): Plot title
            n (int): Number of top characters to plot
        """
        plt.figure(figsize=(10, 6))
        centrality_df.sort_values('centrality', ascending=False).head(n).plot(kind="bar")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def detect_communities(self, graph):
        """
        Detect communities in the character network.
        
        Args:
            graph (networkx.Graph): Graph to analyze
            
        Returns:
            dict: Community assignments
        """
        communities = community_louvain.best_partition(graph)
        nx.set_node_attributes(graph, communities, 'group')
        
        return communities
    
    def visualize_communities(self, graph, output_file="character_communities.html",
                             width="1000px", height="700px",
                             bgcolor='#222222', font_color='white'):
        """
        Visualize communities in the character network.
        
        Args:
            graph (networkx.Graph): Graph with community assignments
            output_file (str): Output HTML file name
            width (str): Width of the visualization
            height (str): Height of the visualization
            bgcolor (str): Background color
            font_color (str): Font color
            
        Returns:
            pyvis.network.Network: Interactive network object
        """
        com_net = Network(notebook=True, width=width, height=height, 
                         bgcolor=bgcolor, font_color=font_color)
        com_net.from_nx(graph)
        com_net.show(output_file)
        
        return com_net
    
    def analyze_character_evolution(self, characters=None):
        """
        Analyze the evolution of character importance across books.
        
        Args:
            characters (list): List of character names to analyze
            
        Returns:
            pd.DataFrame: DataFrame with character centrality evolution
        """
        if not self.graphs:
            raise ValueError("Books must be analyzed first using analyze_all_books()")
        
        # Get degree centrality for all books
        evolution = []
        for book_name, graph in self.graphs.items():
            centrality = nx.degree_centrality(graph)
            evolution.append(centrality)
        
        # Create DataFrame with book names as indices
        degree_evolution_df = pd.DataFrame.from_records(evolution, index=self.graphs.keys())
        
        # If specific characters are requested, filter for them
        if characters:
            try:
                return degree_evolution_df[characters]
            except KeyError as e:
                available = set(degree_evolution_df.columns)
                missing = set(characters) - available
                raise KeyError(f"Characters {missing} not found. Available: {available}")
        
        return degree_evolution_df
    
    def plot_character_evolution(self, characters=None, figsize=(12, 8)):
        """
        Plot the evolution of character importance across books.
        
        Args:
            characters (list): List of character names to plot
            figsize (tuple): Figure size
        """
        evolution_df = self.analyze_character_evolution(characters)
        
        plt.figure(figsize=figsize)
        evolution_df.plot()
        plt.title("Character Importance Evolution Across Books")
        plt.xlabel("Book")
        plt.ylabel("Degree Centrality")
        plt.legend(title="Character")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = CharacterNetworkAnalyzer()
    
    # Load character information
    characters = analyzer.load_characters("characters.csv")
    print(f"Loaded {len(characters)} characters")
    
    # Load book files
    books = analyzer.load_books("data")
    print(f"Found {len(books)} books")
    
    # Analyze a single book
    book = books[0]
    print(f"Analyzing {book.name}...")
    filtered_df, relationship_df, graph = analyzer.analyze_book(book)
    
    # Calculate centrality measures
    centrality = analyzer.calculate_centrality_measures(graph)
    
    # Plot top characters by degree centrality
    analyzer.plot_top_characters(centrality['degree'], title="Top Characters by Degree Centrality")
    
    # Detect communities
    communities = analyzer.detect_communities(graph)
    
    # Create interactive visualization
    analyzer.visualize_interactive_graph(graph, output_file=f"{book.name.split('.')[0]}_network.html")
    
    # Analyze all books
    all_graphs = analyzer.analyze_all_books()
    
    # Plot character evolution
    analyzer.plot_character_evolution(["Geralt", "Ciri", "Yennefer", "Dandelion", "Vesemir"])
