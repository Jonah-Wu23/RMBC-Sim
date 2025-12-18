import pandas as pd

df = pd.read_csv(r"d:\Documents\Bus Project\Sorce code\data\processed\kmb_route_stop_dist.csv")
# Filter for route 68X if needed, though the request says "Route 68X (e.g., every 10 mins)"
# Let's see all routes in the file first
print("Routes in file:", df['route'].unique())

route_68X = df[df['route'] == '68X']
if route_68X.empty:
    print("68X not found, using all stops for bbox")
    bbox = {
        'min_lat': df['lat'].min(),
        'max_lat': df['lat'].max(),
        'min_lon': df['long'].min(),
        'max_lon': df['long'].max()
    }
else:
    bbox = {
        'min_lat': route_68X['lat'].min(),
        'max_lat': route_68X['lat'].max(),
        'min_lon': route_68X['long'].min(),
        'max_lon': route_68X['long'].max()
    }

print("Bounding Box:", bbox)
