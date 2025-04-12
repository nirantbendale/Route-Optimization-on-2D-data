import pandas as pd                                  # For data manipulation and CSV handling
import numpy as np                                  # For numerical computations and array operations
from typing import List, Tuple, Set, Dict          # Import type hints for better code clarity
import math                                        # For mathematical operations like sqrt
import matplotlib.pyplot as plt                    # For creating route visualizations
from queue import PriorityQueue                    # For priority-based queue operations
from dataclasses import dataclass, field           # For creating structured data classes
import time                                        # For timing and performance measurement
from datetime import datetime, timedelta           # For handling dates and time differences
import random                                      # For random selections in route generation
import itertools                                   # For generating combinations and permutations

@dataclass(order=True)                            # Make class comparable based on priority field
class PrioritizedRoute:                           # Class to store route information with priority
    priority: float                               # Priority value for comparing routes
    route: List[dict] = field(compare=False)      # List of stops in the route, not used in comparison
    visited: Set[int] = field(compare=False)      # Set of visited stop IDs, not used in comparison
    total_distance: float = field(compare=False)  # Total route distance, not used in comparison
    
class DeliveryRouteOptimizer:                     # Main class for route optimization
    def __init__(self, csv_file: str):            # Initialize optimizer with data file
        print("Loading data...")                   # Print status message
        self.df = pd.read_csv(csv_file)           # Read data from CSV file into DataFrame
        self.depot = self.df[self.df['Stop-ID'] == 0].iloc[0]  # Extract depot location
        self.deliveries = self.df[(self.df['Stop-ID'] >= 1) & (self.df['Stop-ID'] <= 48)]  # Get delivery points
        self.petrol_stations = self.df[self.df['Stop-ID'] >= 101]  # Get petrol station locations
        
        self.distance_cache = {}                   # Initialize cache for storing distances
        self._precompute_distances()               # Calculate all distances in advance
        
        print(f"Loaded {len(self.deliveries)} delivery points and {len(self.petrol_stations)} petrol stations")  # Status output

    def euclidean_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:  # Calculate distance between points
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)  # Pythagorean theorem

    def _precompute_distances(self):               # Precompute all pairwise distances
        print("Precomputing distances...")         # Status message
        all_points = pd.concat([                   # Combine all points into one DataFrame
            pd.DataFrame([self.depot]),            # Add depot to points
            self.deliveries,                       # Add delivery points
            self.petrol_stations                   # Add petrol stations
        ])
        
        for i, point1 in all_points.iterrows():   # Iterate through first set of points
            for j, point2 in all_points.iterrows():  # Iterate through second set of points
                if i != j:                         # Skip if same point
                    key = (point1['Stop-ID'], point2['Stop-ID'])  # Create unique key for pair
                    self.distance_cache[key] = self.euclidean_distance(  # Store distance in cache
                        (point1['X'], point1['Y']),  # First point coordinates
                        (point2['X'], point2['Y'])   # Second point coordinates
                    )

    def get_distance(self, stop1_id: int, stop2_id: int) -> float:  # Get distance between two stops
        return self.distance_cache.get((stop1_id, stop2_id),         # Try getting distance one way
               self.distance_cache.get((stop2_id, stop1_id)))        # Or try reverse direction

    def calculate_route_distance(self, route: List[dict]) -> float:  # Calculate total route distance
        total = 0                                  # Initialize total distance
        for i in range(len(route)-1):             # Loop through consecutive pairs of stops
            total += self.get_distance(route[i]['Stop-ID'], route[i+1]['Stop-ID'])  # Add distance between stops
        return total                              # Return total distance

    def two_opt_swap(self, route: List[dict], i: int, j: int) -> List[dict]:  # Perform 2-opt swap operation
        new_route = route[:i]                     # Keep first part of route
        new_route.extend(reversed(route[i:j + 1]))  # Reverse middle segment
        new_route.extend(route[j + 1:])           # Keep last part of route
        return new_route                          # Return new route

    def two_opt_local_search(self, route: List[dict], max_iterations: int = 10000) -> Tuple[List[dict], float]:  # Improve route with 2-opt
        best_route = route                        # Store best route found so far
        best_distance = self.calculate_route_distance(route)  # Calculate initial distance
        improved = True                           # Flag to track improvements
        iteration = 0                             # Initialize iteration counter
        
        while improved and iteration < max_iterations:  # Continue while improving and within limit
            improved = False                      # Reset improvement flag
            iteration += 1                        # Increment iteration counter
            
            for i in range(1, len(route) - 2):   # Try all possible first positions
                for j in range(i + 1, len(route) - 1):  # Try all possible second positions
                    new_route = self.two_opt_swap(best_route, i, j)  # Try swapping segments
                    new_distance = self.calculate_route_distance(new_route)  # Calculate new distance
                    
                    if new_distance < best_distance:  # If improvement found
                        best_route = new_route    # Update best route
                        best_distance = new_distance  # Update best distance
                        improved = True           # Mark as improved
                        print(f"2-opt improved distance to: {best_distance:.2f}")  # Report improvement
                        break                     # Break inner loop
                
                if improved:                      # If improved in this iteration
                    break                         # Break outer loop
        
        return best_route, best_distance          # Return best route and its distance

    def generate_initial_solutions(self, num_solutions: int = 50) -> List[Tuple[List[dict], float]]:  # Generate multiple starting solutions
        solutions = []                            # List to store all solutions
        print(f"Generating {num_solutions} initial solutions...")  # Status message
        
        for i in range(num_solutions):           # Generate specified number of solutions
            current = self.depot.to_dict()        # Start at depot
            unvisited = set(self.deliveries['Stop-ID'])  # Set of unvisited stops
            route = [current]                     # Initialize route with depot
            total_distance = 0                    # Initialize total distance

            while unvisited:                      # While there are unvisited stops
                current_id = current['Stop-ID']   # Get current stop ID
                candidates = [(next_id, self.get_distance(current_id, next_id))  # Get distances to all candidates
                            for next_id in unvisited]
                candidates.sort(key=lambda x: x[1])  # Sort by distance
                
                top_n = min(10, len(candidates))  # Get number of candidates to consider
                next_id = random.choice(candidates[:top_n])[0]  # Randomly select from top candidates
                
                next_stop = self.deliveries[self.deliveries['Stop-ID'] == next_id].iloc[0].to_dict()  # Get next stop details
                route.append(next_stop)           # Add stop to route
                total_distance += self.get_distance(current_id, next_id)  # Add distance to total
                unvisited.remove(next_id)         # Remove from unvisited
                current = next_stop               # Update current position

            total_distance += self.get_distance(current['Stop-ID'], 0)  # Add distance back to depot
            route.append(self.depot.to_dict())    # Add depot to end of route
            
            improved_route, improved_distance = self.two_opt_local_search(route)  # Improve route
            solutions.append((improved_route, improved_distance))  # Add to solutions
            
            if (i + 1) % 5 == 0:                 # Every 5 solutions
                print(f"Generated {i + 1}/{num_solutions} initial solutions")  # Report progress
        
        return solutions                          # Return all solutions

    def optimize_route(self, time_limit_minutes: int = 15) -> Tuple[List[dict], float]:  # Main optimization function
        print("\nStarting route optimization...")  # Start message
        start_time = datetime.now()               # Record start time
        end_time = start_time + timedelta(minutes=time_limit_minutes)  # Calculate end time

        print("Generating initial solutions...")   # Status message
        initial_solutions = self.generate_initial_solutions(num_solutions=50)  # Generate starting solutions
        best_solution, best_distance = min(initial_solutions, key=lambda x: x[1])  # Find best initial solution
        print(f"Best initial solution distance: {best_distance:.2f}")  # Report best initial distance

        improvement_rounds = 3                    # Number of improvement rounds
        for round in range(improvement_rounds):   # For each improvement round
            print(f"\nImprovement round {round + 1}/{improvement_rounds}")  # Report round number
            improved_solution, improved_distance = self.two_opt_local_search(  # Try to improve solution
                best_solution, max_iterations=20000)
            
            if improved_distance < best_distance:  # If better solution found
                best_solution = improved_solution  # Update best solution
                best_distance = improved_distance  # Update best distance
                print(f"Found better solution! Distance: {best_distance:.2f}")  # Report improvement

        print(f"\nOptimization completed!")       # Completion message
        print(f"Best distance found: {best_distance:.2f}")  # Report final distance
        
        print("\nOptimizing refueling stop positions...")  # Status message
        final_route = self.insert_refueling_stops(best_solution)  # Add refueling stops
        final_distance = self.calculate_route_distance(final_route)  # Calculate final distance
        
        return final_route, final_distance        # Return optimized route and distance

    def insert_refueling_stops(self, route: List[dict]) -> List[dict]:  # Insert refueling stops
        new_route = [route[0]]                    # Start with depot
        deliveries = [stop for stop in route if 1 <= stop['Stop-ID'] <= 48]  # Get delivery stops

        group_sizes = [                           # Different grouping patterns to try
            [8, 9, 9, 9, 9, 4],                  # Pattern 1
            [9, 9, 9, 9, 9, 3],                  # Pattern 2
            [9, 8, 9, 9, 9, 4],                  # Pattern 3
            [9, 9, 8, 9, 9, 4],                  # Pattern 4
            [9, 9, 9, 8, 9, 4]                   # Pattern 5
        ]

        best_route = None                         # Store best route found
        best_distance = float('inf')              # Initialize best distance as infinity

        for sizes in group_sizes:                 # Try each grouping pattern
            current_route = [route[0]]            # Start with depot
            start = 0                             # Start index for grouping
            groups = []                           # Store delivery groups
            
            for size in sizes:                    # Split deliveries into groups
                end = start + size                # Calculate end index
                groups.append(deliveries[start:end])  # Add group
                start = end                       # Update start index

            for i in range(len(groups)):          # Process each group
                current_route.extend(groups[i])    # Add current group to route
                
                if i < 5:                         # If not last group
                    prev_stop = current_route[-1]  # Get last stop
                    next_stop = groups[i+1][0] if i+1 < len(groups) else route[-1]  # Get next stop
                    
                    best_station = None            # Store best station found
                    min_detour = float('inf')      # Initialize minimum detour
                    
                    for _, station in self.petrol_stations.iterrows():  # Try each station
                        detour = (                 # Calculate detour distance
                            self.get_distance(prev_stop['Stop-ID'], station['Stop-ID']) +
                            self.get_distance(station['Stop-ID'], next_stop['Stop-ID'])
                        )
                        direct = self.get_distance(prev_stop['Stop-ID'], next_stop['Stop-ID'])  # Direct distance
                        added_distance = detour - direct  # Calculate added distance
                        
                        if added_distance < min_detour:  # If better option found
                            min_detour = added_distance  # Update minimum detour
                            best_station = station.to_dict()  # Update best station
                    
                    if best_station:              # If station found
                        current_route.append(best_station)  # Add best station
                    else:                         # If no station found
                        current_route.append(self.petrol_stations.iloc[0].to_dict())  # Add first station

            current_route.append(route[-1])       # Add final depot
            current_distance = self.calculate_route_distance(current_route)  # Calculate total distance

            if current_distance < best_distance:   # If better arrangement found
                best_distance = current_distance   # Update best distance
                best_route = current_route         # Update best route
                print(f"Found better refueling arrangement: {best_distance:.2f}")  # Report improvement

        return best_route                         # Return best route found

    def display_route_with_distances(self, route: List[dict], total_distance: float) -> None:  # Display route details
        print("\nRoute sequence with distances between stops:")  # Header
        for i in range(len(route) - 1):          # For each pair of stops
            current_stop = route[i]               # Get current stop
            next_stop = route[i + 1]              # Get next stop
            
            distance = self.get_distance(current_stop['Stop-ID'], next_stop['Stop-ID'])  # Calculate distance
            
            stop_type = "Depot" if current_stop['Stop-ID'] == 0 else \
                       "Petrol" if current_stop['Stop-ID'] >= 101 else "Delivery"
            print(f"{i+1}. {stop_type} {current_stop['Stop-ID']} -> "  # Print stop information
                  f"{next_stop['Stop-ID']}: {distance:.2f} units")
        
        print(f"\nTotal route distance: {total_distance:.2f} units")  # Print total distance

    def plot_route(self, route: List[dict], total_distance: float) -> None:  # Create route visualization
        plt.figure(figsize=(12, 10))             # Create new figure with specified size
        
        for i in range(len(route)-1):            # For each segment in route
            start = route[i]                      # Get start point of segment
            end = route[i+1]                      # Get end point of segment
            
            plt.plot([start['X'], end['X']], [start['Y'], end['Y']],   # Draw line between points
                    'k-', linewidth=0.5, alpha=0.5)                     # Set line style properties
            
            mid_x = start['X'] + 0.6 * (end['X'] - start['X'])         # Calculate arrow midpoint x
            mid_y = start['Y'] + 0.6 * (end['Y'] - start['Y'])         # Calculate arrow midpoint y
            plt.arrow(mid_x, mid_y,                                     # Draw direction arrow
                     (end['X'] - start['X'])/10, (end['Y'] - start['Y'])/10,  # Set arrow direction
                     head_width=0.3, head_length=0.5, fc='black', ec='black',  # Set arrow properties
                     alpha=0.5)                                         # Set arrow transparency
        
        plt.scatter(self.depot['X'], self.depot['Y'], c='red', s=50, zorder=5)  # Plot depot as red dot
        plt.annotate(f"0", (self.depot['X'], self.depot['Y']),         # Add depot label
                    xytext=(5, 5), textcoords='offset points',          # Set label position
                    color='red', fontsize=8, fontweight='bold')         # Set label style
        
        for _, delivery in self.deliveries.iterrows():                  # Plot all delivery points
            plt.scatter(delivery['X'], delivery['Y'], c='blue', s=30, zorder=4)  # Plot as blue dots
            plt.annotate(f"{int(delivery['Stop-ID'])}",                 # Add stop ID label
                        (delivery['X'], delivery['Y']),                  # Label position
                        xytext=(5, 5), textcoords='offset points',      # Label offset
                        color='blue', fontsize=8)                       # Label style
        
        for _, station in self.petrol_stations.iterrows():             # Plot all petrol stations
            plt.scatter(station['X'], station['Y'], c='none',          # Plot as circles
                      edgecolor='green', s=50, linewidth=1, zorder=3)  # Set circle style
            plt.annotate(f"{int(station['Stop-ID'])}",                 # Add station ID label
                        (station['X'], station['Y']),                   # Label position
                        xytext=(5, 5), textcoords='offset points',     # Label offset
                        color='green', fontsize=8)                     # Label style
        
        plt.grid(True, color='#E0E0E0', linestyle='-', linewidth=0.5)  # Add grid to plot
        plt.xlim(-5, 30)                                               # Set x-axis limits
        plt.ylim(-5, 25)                                               # Set y-axis limits
        plt.title(f'Route distance = {total_distance:.2f}', pad=10)    # Add title with distance
        plt.gca().set_aspect('equal', adjustable='box')                # Set equal aspect ratio
        
        plt.savefig('route_map.png', dpi=300, bbox_inches='tight')    # Save high-resolution image
        plt.close()                                                    # Close the plot

def main():                                                           # Main function
    try:                                                             # Error handling block
        print("Initializing Route Optimizer...")                     # Start message
        optimizer = DeliveryRouteOptimizer('data.csv')              # Create optimizer instance
        
        print("\nStarting route optimization...")                    # Status message
        route, total_distance = optimizer.optimize_route(time_limit_minutes=15)  # Run optimization
        
        print("\nGenerating route details...")                      # Status message
        optimizer.display_route_with_distances(route, total_distance)  # Display route details
        
        print("\nCreating route visualization...")                  # Status message
        optimizer.plot_route(route, total_distance)                 # Create visualization
        print("Route map saved as 'route_map.png'")                # Confirmation message
        
        print("\nRoute Summary:")                                   # Summary header
        print(f"Total stops: {len(route)}")                        # Print total stops
        delivery_stops = sum(1 for stop in route if 1 <= stop['Stop-ID'] <= 48)  # Count delivery stops
        petrol_stops = sum(1 for stop in route if stop['Stop-ID'] >= 101)        # Count petrol stops
        print(f"Delivery stops: {delivery_stops}")                  # Print delivery stops count
        print(f"Petrol stops: {petrol_stops}")                     # Print petrol stops count
        print(f"Total distance: {total_distance:.2f} units")       # Print total distance
        
    except Exception as e:                                         # Exception handling
        print(f"Error: {str(e)}")                                 # Print error message
        raise                                                     # Re-raise exception

if __name__ == "__main__":                                        # Script entry point
    main()                                                        # Run main function