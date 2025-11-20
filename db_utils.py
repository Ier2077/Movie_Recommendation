def movie_finder(title_arr,db_collection):
    if not title_arr:
        return []
    
    title_lower = [title.lower() for title in title_arr]
    try:
        # Query the database for the given titles
        results = db_collection.get(
            
            where = {"title_lower": {"$in": title_lower}},
            include = []
        )
        
        found_movies = results["ids"]
        print(f"Found {len(found_movies)} ids in the database.")
        return found_movies
        
        
    except Exception as e:
        print(f"Error querying the database: {e}")
        return []
    
    