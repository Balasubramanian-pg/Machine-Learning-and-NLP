# Character Network Analysis - Detailed Code Notes

This document provides a comprehensive line-by-line explanation of the Character Network Analysis code.

## Overview

The code creates a `CharacterNetworkAnalyzer` class for analyzing character relationships in literary texts using Natural Language Processing (NLP) and graph theory. It extracts characters, identifies their interactions, and visualizes the resulting social networks.

## Imports Section

```python
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
```

- `os`: Used for file system operations (reading book files)
- `re`: For regular expressions to clean character names
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `spacy`: NLP library for named entity recognition
- `displacy`: SpaCy's visualization module
- `networkx`: For creating and analyzing networks/graphs
- `matplotlib.pyplot`: For plotting charts and graphs
- `pyvis.network`: For interactive network visualizations
- `community_louvain`: For community detection in networks

## Class Definition

```python
class CharacterNetworkAnalyzer:
    """Analyzes character networks in text using NLP and graph theory."""
```

This defines the main class that will encapsulate all functionality.

## Initialization Method

```python
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
```

- Takes an optional parameter `language_model` that defaults to SpaCy's small English model
- Tries to load the specified language model
- If the model isn't installed, automatically downloads it
- Initializes three empty containers:
  - `characters_df`: Will hold character information
  - `book_docs`: Dictionary to store processed book documents
  - `graphs`: Dictionary to store character networks

## Loading Characters

```python
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
```

- Loads character names from a CSV file
- Cleans character names by:
  - Removing text within parentheses using regex
  - Stripping whitespace
- Creates a new column `character_firstname` with just the first name
- Returns the processed DataFrame

## Loading Books

```python
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
```

- Scans the specified directory for text files
- Uses list comprehension to filter only files with '.txt' in their name
- Sorts the books by filename (useful for chronological order)
- Returns the sorted list of book files

## Processing a Book

```python
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
```

- Takes a book file path
- Extracts the filename using `os.path.basename`
- Reads the entire file content
- Processes the text with SpaCy's NLP pipeline (tokenization, part-of-speech tagging, named entity recognition)
- Stores the processed document in the `book_docs` dictionary
- Returns the processed document

## Visualizing Named Entities

```python
def visualize_entities(self, doc, start=0, end=2000):
    """
    Visualize named entities in a document.
    
    Args:
        doc (spacy.Doc): Document to visualize
        start (int): Start index
        end (int): End index
    """
    return displacy.render(doc[start:end], style="ent", jupyter=True)
```

- Uses SpaCy's displacy to visualize named entities
- Takes only a slice of the document (`start` to `end`) to avoid overloading the browser
- Returns a visualization highlighting different entity types (people, locations, organizations, etc.)

## Extracting Entities per Sentence

```python
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
```

- Creates an empty list to store sentence-entity pairs
- Loops through each sentence in the document
- For each sentence, extracts all named entities using SpaCy's entity recognition
- Adds a dictionary with the sentence and its entities to the list
- Returns a DataFrame with two columns: `sentence` and `entities`

## Filtering Character Entities

```python
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
```

- Checks if characters have been loaded
- Defines a helper function to filter entities that match character names
- Adds a new column `character_entities` with only the filtered character entities
- Removes sentences that don't have any character entities
- Simplifies character names to first names only
- Returns the filtered DataFrame

## Creating Character Relationships

```python
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
```

- Creates an empty list to store character relationships
- For each sentence index:
  - Defines a window of sentences (of size `window_size`)
  - Extracts all character entities in this window
  - Flattens the list of character lists with `sum(..., [])`
  - Removes consecutive duplicates (same character mentioned multiple times)
  - If there are at least 2 unique characters in the window:
    - Creates relationships between consecutive characters
    - Adds each relationship with a value of 1
- Creates a DataFrame from the relationships
- If the DataFrame is not empty:
  - Sorts source-target pairs to normalize the direction
  - Groups by source-target pairs and sums the values to count relationship occurrences
- Returns the relationship DataFrame

## Creating a Network Graph

```python
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
```

- Creates a NetworkX graph from the relationship DataFrame
- Uses the `from_pandas_edgelist` function to:
  - Set the source column as the source nodes
  - Set the target column as the target nodes
  - Use the value column as edge attributes
  - Create an undirected graph
- Returns the created graph

## Analyzing a Book

```python
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
```

- Extracts the book name from the file path
- Processes the book to extract named entities
- Extracts entities for each sentence
- Filters to keep only character entities
- Creates character relationships
- Builds a network graph
- Stores the graph in the `graphs` dictionary
- Returns the intermediate DataFrames and the final graph

## Analyzing All Books

```python
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
```

- Checks if books have been loaded
- Loops through each book file
- Analyzes each book using the `analyze_book` method
- Returns the dictionary of graphs for all books

## Visualizing a Graph

```python
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
```

- Creates a matplotlib figure with the specified size
- Uses the Kamada-Kawai layout algorithm to position nodes
- Draws the graph with:
  - Node labels visible
  - Nodes colored skyblue
  - Edges using a blue color map
  - Node positions from the layout algorithm
- Displays the graph

## Creating an Interactive Graph Visualization

```python
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
```

- Creates a pyvis Network object with custom styling
- Sets node sizes based on their degree (number of connections)
- Converts the NetworkX graph to a pyvis network
- Saves the interactive visualization as an HTML file
- Returns the network object

## Calculating Centrality Measures

```python
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
```

- Calculates three types of centrality measures:
  - Degree centrality: How many direct connections a character has
  - Betweenness centrality: How often a character appears on shortest paths between other characters
  - Closeness centrality: How close a character is to all other characters
- Converts each measure to a DataFrame for easy analysis
- Adds the centrality measures as node attributes in the graph
- Returns a dictionary with all three DataFrames

## Plotting Top Characters

```python
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
```

- Creates a matplotlib figure
- Sorts the centrality DataFrame by the centrality value
- Takes the top `n` characters
- Creates a bar plot
- Adds a title
- Adjusts layout for better readability
- Displays the plot

## Detecting Communities

```python
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
```

- Uses the Louvain algorithm for community detection
- Returns a dictionary mapping each character to its community
- Adds community assignments as a 'group' attribute to nodes

## Visualizing Communities

```python
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
```

- Creates a pyvis Network object with custom styling
- Converts the NetworkX graph to a pyvis network
- The graph should already have community assignments as node attributes
- Saves the visualization as an HTML file
- Returns the network object

## Analyzing Character Evolution

```python
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
```

- Checks if books have been analyzed
- Creates an empty list to store centrality dictionaries
- For each book graph:
  - Calculates degree centrality for all characters
  - Adds the centrality dictionary to the list
- Creates a DataFrame from the list of dictionaries
- Sets book names as the index
- If specific characters are requested:
  - Tries to filter the DataFrame to include only those characters
  - If any characters are not found, raises an error with helpful information
- Returns the evolution DataFrame

## Plotting Character Evolution

```python
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
```

- Gets the character evolution DataFrame
- Creates a matplotlib figure with the specified size
- Plots the evolution DataFrame
- Adds a title, axis labels, and legend
- Adds a grid for better readability
- Displays the plot

## Example Usage

```python
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
```

This example code shows how to use the `CharacterNetworkAnalyzer` class:

1. Initialize the analyzer
2. Load character information from a CSV file
3. Load book files from a directory
4. Analyze a single book
5. Calculate centrality measures for the book
6. Plot the top characters by degree centrality
7. Detect and visualize character communities
8. Create an interactive visualization of the character network
9. Analyze all books
10. Plot the evolution of specific characters across the book series

## Summary

This code provides a comprehensive framework for analyzing character networks in literary texts. It:

1. Extracts named entities from text using SpaCy
2. Filters entities to identify character mentions
3. Creates relationships between characters based on co-occurrence
4. Builds network graphs to represent character interactions
5. Calculates various centrality measures to identify important characters
6. Detects communities of closely related characters
7. Analyzes how character importance evolves across multiple books
8. Provides both static and interactive visualizations

The modular class-based approach makes it easy to reuse and extend this code for different literary works or to add new analysis methods.
