#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import tkinter
import tkinter.messagebox
from tkintermapview import TkinterMapView

import pandas as pd

from pyswip import Prolog

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download as nltkdl
import string


# In[2]:


nltkdl('wordnet')
nltkdl('stopwords')
nltkdl('punkt')


# In[3]:


def sanatize(string):
    return string.lower().replace(" ", "_").replace("-", "_").replace('\'', '').replace('.', '')


# In[4]:


def find_related_words(word):
    synonyms = set()
    synonyms.add(word)
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())

    return synonyms


# In[5]:


dest_df = pd.read_csv('Destinations.csv')
dest_df.head(10)


# In[6]:


features = list(map(sanatize, dest_df.columns))
dest_features = {}
values = dest_df.values
for val in values:
    dest_features[sanatize(val[0])] = {}
    for i in range(1, len(features)):
        dest_features[sanatize(val[0])][features[i]] = sanatize(val[i])


# In[7]:


dest_features


# In[8]:


map_df = pd.read_csv('Adjacency_matrix.csv', index_col="Destinations")
map_df.head()


# In[9]:


categories = {}
for col in dest_df.columns:
    categories[sanatize(col)] = find_related_words(sanatize(col))
    categories[sanatize(col)].add(sanatize(col))
    
informations = {}
for col in dest_df.columns:
    informations[sanatize(col)] = {}
    for val in dest_df[col]:
        informations[sanatize(col)][sanatize(val)] = {}

for category in informations:
    for val in informations[category]:
        informations[category][val] = find_related_words(val)


# In[10]:


def point_location(location, features):
    point = 0
    for feature in features:
        if dest_features[location][feature] in features[feature]:
            point += 1
    return point


# In[11]:


def result_exists(res_list):
    return len(list(res_list)) > 0


# In[12]:


def get_tours(cities):
    if len(cities) == 0:
        raise Exception("No cities were found. Enter a more specific input.")

    if len(cities) > 5:
        raise Exception("Too many cities were found. Enter a more specific input.")

    paths = []

    if len(cities) == 5:
        for c in cities:
            paths.extend(trial([x for x in cities if c != x]))
    else:
        paths.extend(trial(cities))
        
    return replace_variables(paths)


# In[13]:


def trial(cities):
    varc = 0
    removec = 0

    total_ok_paths = []

    while len(cities) - removec > 0:
        if removec == 0:
            ok_paths = check_cities(cities)

            if len(ok_paths) > 0:
                total_ok_paths.append(ok_paths)
        elif removec == 1:
            for c in cities:
                ok_paths = check_cities([city for city in cities if city != c])

                if len(ok_paths) > 0:
                    total_ok_paths.append(ok_paths)
        elif removec == 2:
            for c1 in cities:
                for c2 in cities:
                    if c1 == ct2:
                        continue
    
                    ok_paths = check_cities([city for city in cities if city != c1 and city != c2])

                    if len(ok_paths) > 0:
                        total_ok_paths.append(ok_paths)

        if len(total_ok_paths) > 0:
            return total_ok_paths

        removec += 1

    return []


# In[14]:


def check_cities(cities):
    varc = 0
    total_ok_paths = []

    while varc + len(cities) <= 4:
        ok_paths = check_cities_varc(cities, varc)

        if len(ok_paths) > 0:
            total_ok_paths = ok_paths
    
        if len(total_ok_paths) > 0:
            return total_ok_paths
    
        varc += 1
    return []


# In[15]:


def check_cities_varc(cities, varc):
    paths = get_permutation(cities, varc)
    
    ok_paths = []
    for p in paths:
        res = check_connection(p)
        if res is not None:
            ok_paths.append((p, res))


    return ok_paths


# In[16]:


def get_permutation(cities, varc = 0):
    res = []
    if len(cities) == 1:
        return [cities]

    for i in range(len(cities)):
        for j in range(len(cities)):
            if i == j:
                continue

            src = cities[i]
            dest = cities[j]
            cnt = len(cities)

            rem = [x for xi, x in enumerate(cities) if xi != i and xi != j]

            if cnt == 2:
                if varc == 0:
                    res.append((src, dest))
                elif varc == 1:
                    res.append((src, 'X1', dest))
                elif varc == 2:
                    res.append((src, 'X1', 'X2', dest))
            if cnt == 3:
                if varc == 0:
                    res.append((src, dest))
                elif varc == 1:
                    res.append((src, 'X1', rem[0], dest))
                    res.append((src, rem[0], 'X1', dest))
            if cnt == 4:
                if varc == 0:
                    res.append((src, rem[0], rem[1], dest))
                    res.append((src, rem[1], rem[0], dest))
                

    return res


# In[17]:


def check_connection(cities):
    if len(cities) <= 1:
        return [{}]

    prev = None
    query = ""
    for i, ct in enumerate(cities):
        if i == 0:
            prev = ct
            continue

        query += f"connected({prev}, {ct})"
        prev = ct
        if i != len(cities) - 1:
            query += ", "
        
    result = prolog.query(query)

    res_list = list(result)
    if result_exists(res_list):
        return res_list

    return None


# In[18]:


def replace_variables(res):
    results = []
    for city_set in res:
        for type in city_set:
            pattern = str(type[0])[1:-1]
            for object in type[1]:
                val = '%s' % pattern
                for var in object:
                    val = val.replace(var, object[var])
                results.append(val) 
    return results


# In[19]:


def delete_punctuation(text):
    # Remove punctuation using the string module
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator)
    return text_without_punct

def delete_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return [word for word in word_tokens if word.lower() not in stop_words]

def keywords_text(text):
    return delete_stopwords(delete_punctuation(text))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# In[20]:


class App(tkinter.Tk):

    APP_NAME = "map_view_demo.py"
    WIDTH = 800
    HEIGHT = 750  # This is now the initial size, not fixed.

    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)

        self.title(self.APP_NAME)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        # Configure the grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Text area and submit button combined row
        self.grid_rowconfigure(1, weight=4)  # Map row

        # Upper part: Text Area and Submit Button
        self.text_area = tkinter.Text(self, height=5)  # Reduced height for text area
        self.text_area.grid(row=0, column=0, pady=(10, 0), padx=10, sticky="nsew")

        self.submit_button = tkinter.Button(self, text="Submit", command=self.process_text)
        self.submit_button.grid(row=0, column=0, pady=(0, 10), padx=10, sticky="se")  # Placed within the same cell as text area

        # Lower part: Map Widget
        self.map_widget = TkinterMapView(self)
        self.map_widget.grid(row=1, column=0, sticky="nsew")

        self.marker_list = []  # Keeping track of markers
        self.marker_path = None


    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)

        self.title(self.APP_NAME)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        # Configure the grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Text area can expand/contract.
        self.grid_rowconfigure(1, weight=0)  # Submit button row; doesn't need to expand.
        self.grid_rowconfigure(2, weight=3)  # Map gets the most space.

        # Upper part: Text Area and Submit Button
        self.text_area = tkinter.Text(self)
        self.text_area.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
        
        self.submit_button = tkinter.Button(self, text="Submit", command=self.process_text)
        self.submit_button.grid(row=1, column=0, pady=10, sticky="ew")

        # Lower part: Map Widget
        self.map_widget = TkinterMapView(self)
        self.map_widget.grid(row=2, column=0, sticky="nsew")

        self.marker_list = []  # Keeping track of markers

    def check_connections(self, results):
        locations = []
        for result in results:
            city = result["City"]
            locations.append(city)
        return locations

    def process_text(self):
        """Extract locations from the text area and mark them on the map."""
        text = self.text_area.get("1.0", "end-1c")  # Get text from text area
        features = self.extract_features(text)  # Extract locations (you may use a more complex method here)

        properties = ""
        for i, col in enumerate(dest_df.columns):
            if i == 0:
                continue
            properties += f'{list(features[sanatize(col)])[0] if sanatize(col) in features else "_"}'

            if i != len(dest_df.columns) - 1:
                properties += ", "
        query = f"destination(City, {properties})"

        print(f"{query=}")
        
        results = list(prolog.query(query))
        locations = self.check_connections(results)
        print(f"{locations=}")

        tours = get_tours(locations[0:4])
        best_path = self.evaluate(tours, features)
        print("Best Path:", best_path)
        self.mark_locations(best_path[0])

    def mark_locations(self, locations):
        """Mark extracted locations on the map."""
        for address in locations:
            marker = self.map_widget.set_address(address, marker=True)
            if marker:
                self.marker_list.append(marker)
        self.connect_marker()
        self.map_widget.set_zoom(1)  # Adjust as necessary, 1 is usually the most zoomed out


    def connect_marker(self):
        position_list = []

        for marker in self.marker_list:
            position_list.append(marker.position)

        if hasattr(self, 'marker_path') and self.marker_path is not None:
            self.map_widget.delete(self.marker_path)

        if len(position_list) > 0:
            self.marker_path = self.map_widget.set_path(position_list)

    def extract_features(self, text):
        keywords = keywords_text(text)
        features = {}
        for category in categories:
            for value in categories[category]:
                if value in keywords:
                    features[category] = set()  
                    
        for category in features:
            for value in informations[category]:
                for related in informations[category][value]:
                    if related in keywords:
                        features[category].add(value)
    
        return features

    def point_location(self, location, features):
        point = 0
        for feature in features:
            if dest_features[location][feature] in features[feature]:
                point += 1
        return point

    def evaluate(self, paths, features):
        best_path = ("", 0)
        for path in paths:
            point = 0
            print(f"path = {path}")
            arr_path = [x[1:-1] for x in path.split(', ')]
            for city in arr_path:
                point += self.point_location(city, features)
            if best_path[1] <= point:
                best_path = (arr_path, point)
        return best_path
                
    def start(self):
        self.mainloop()


# In[21]:


prolog = Prolog()


# In[ ]:


prolog.retractall("destination(_, _, _, _, _, _, _, _, _, _, _, _, _)")
assertions = []
for data in dest_df.values:
    assert_str = ''
    for i in range(len(data)):
        if i != len(data) - 1:
            assert_str += '\'' + sanatize(data[i]) + '\', '
        else:
            languages = data[i].split(", ")
            for lang in languages:
                temp = assert_str
                temp += '\'' + sanatize(lang) + '\''
                assertions.append(temp)

for assert_str in assertions:            
    prolog.assertz(f"destination({assert_str})")


# In[23]:


prolog.retractall("directly_connected(_, _)")
prolog.retractall("connected(_, _)")

all_cities = map_df.index
visited = set()

for ct1 in all_cities:
    for ct2 in all_cities:
        if ct2 == ct1:
            continue

        if map_df[ct1][ct2] and (ct1, ct2) not in visited:
            prolog.assertz(f"directly_connected('{sanatize(ct1)}', '{sanatize(ct2)}')")

            visited.add((ct1, ct2))
            visited.add((ct2, ct1))

prolog.assertz("connected(X, Y) :- directly_connected(X, Y)")
prolog.assertz("connected(X, Y) :- directly_connected(Y, X)")


# In[ ]:


if __name__ == "__main__":
    app = App()
    app.start()


# In[ ]:





# In[ ]:




