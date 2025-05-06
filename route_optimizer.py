"""
Logistics Route Optimizer - MVP
------------------------------
This script demonstrates a simple logistics route optimization using:
1. Distance matrix calculation
2. Vehicle routing problem (VRP) solver using Google OR-Tools
3. Carbon emissions estimation
4. Route visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from folium import plugins
import random
import math

class LogisticsRouteOptimizer:
    def __init__(self, warehouse_location=None, num_delivery_points=10):
        """Initialize the route optimizer with locations"""
        # Set warehouse location (default: center of grid)
        self.warehouse_location = warehouse_location or (0, 0)
        
        # Generate random delivery points if not provided
        self.delivery_points = self._generate_delivery_points(num_delivery_points)
        
        # Create distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Vehicle parameters
        self.num_vehicles = 3
        self.vehicle_capacity = 15  # packages per vehicle
        
        # Demand for each location (random between 1-5 packages)
        self.demands = [0] + [random.randint(1, 5) for _ in range(num_delivery_points)]
        
        # Carbon emission factors (kg CO2 per km)
        self.emission_factors = {
            'small_van': 0.15,
            'medium_van': 0.21,
            'large_van': 0.28
        }
        
    def _generate_delivery_points(self, num_points):
        """Generate random delivery points within a grid"""
        # Generate points in roughly circular pattern around warehouse
        points = []
        for _ in range(num_points):
            # Random angle and distance
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(5, 20)  # km from warehouse
            
            # Convert to x, y coordinates
            x = self.warehouse_location[0] + distance * math.cos(angle)
            y = self.warehouse_location[1] + distance * math.sin(angle)
            
            points.append((x, y))
        
        return points
    
    def _calculate_distance_matrix(self):
        """Calculate Euclidean distance matrix between all points"""
        # Combine warehouse and delivery points
        all_points = [self.warehouse_location] + self.delivery_points
        num_points = len(all_points)
        
        # Create distance matrix
        distance_matrix = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            for j in range(i, num_points):
                if i == j:
                    distance = 0
                else:
                    # Euclidean distance
                    distance = math.sqrt(
                        (all_points[i][0] - all_points[j][0])**2 + 
                        (all_points[i][1] - all_points[j][1])**2
                    )
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric
        
        return distance_matrix
    
    def solve_vrp(self):
        """Solve the Vehicle Routing Problem using OR-Tools"""
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix),
            self.num_vehicles,
            0  # depot index
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node] * 1000)  # Convert to meters (int)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.vehicle_capacity] * self.num_vehicles,  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add distance constraint
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            100000,  # maximum distance per vehicle (100 km in meters)
            True,  # start cumul to zero
            dimension_name
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 10
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        # Extract results
        if solution:
            return self._process_solution(manager, routing, solution)
        else:
            print("No solution found!")
            return None
    
    def _process_solution(self, manager, routing, solution):
        """Process the solution to extract routes and metrics"""
        routes = []
        total_distance = 0
        total_packages = 0
        
        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                
                # Add load
                route_load += self.demands[node_index]
                
                # Add distance to next stop
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id) / 1000  # Convert back to km
            
            # Add depot as last node
            route.append(manager.IndexToNode(index))
            
            # Collect route information
            vehicle_type = list(self.emission_factors.keys())[vehicle_id % len(self.emission_factors)]
            emissions = route_distance * self.emission_factors[vehicle_type]
            
            route_info = {
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'route': route,
                'distance': route_distance,
                'packages': route_load,
                'emissions': emissions
            }
            
            routes.append(route_info)
            total_distance += route_distance
            total_packages += route_load
        
        # Calculate total emissions
        total_emissions = sum(r['emissions'] for r in routes)
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_packages': total_packages,
            'total_emissions': total_emissions
        }
    
    def plot_routes(self, solution, save_file=None):
        """Plot the routes using Matplotlib"""
        routes = solution['routes']
        
        # Combine all locations
        all_locations = [self.warehouse_location] + self.delivery_points
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot all delivery points
        x = [loc[0] for loc in self.delivery_points]
        y = [loc[1] for loc in self.delivery_points]
        plt.scatter(x, y, c='blue', s=50, label='Delivery Points')
        
        # Plot warehouse
        plt.scatter([self.warehouse_location[0]], [self.warehouse_location[1]], 
                   c='red', s=200, marker='*', label='Warehouse')
        
        # Plot routes
        colors = ['green', 'purple', 'orange', 'brown', 'pink']
        for i, route_info in enumerate(routes):
            route = route_info['route']
            color = colors[i % len(colors)]
            
            # Extract route coordinates
            route_x = [all_locations[j][0] for j in route]
            route_y = [all_locations[j][1] for j in route]
            
            # Plot route
            plt.plot(route_x, route_y, c=color, linewidth=2, 
                    label=f"Vehicle {i+1} - {route_info['packages']} packages")
            
            # Add direction arrows
            for j in range(len(route) - 1):
                plt.annotate('', 
                            xy=(route_x[j+1], route_y[j+1]),
                            xytext=(route_x[j], route_y[j]),
                            arrowprops=dict(arrowstyle='->', color=color))
        
        # Add labels
        for i, (x, y) in enumerate(self.delivery_points, 1):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Optimized Delivery Routes\nTotal Distance: {solution["total_distance"]:.2f} km, Emissions: {solution["total_emissions"]:.2f} kg CO2')
        plt.xlabel('Distance (km)')
        plt.ylabel('Distance (km)')
        plt.grid(True)
        plt.legend()
        
        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()
    
    def create_map(self, solution, save_file=None):
        """Create an interactive map with the routes using Folium"""
        # This is a simplified version - in a real project you'd use real geocoordinates
        
        # Convert our abstract coordinates to approximate lat/lon
        # We'll center around New York City as an example
        base_lat, base_lon = 40.7128, -74.0060
        
        # Scale factor to convert our units to approximate degrees
        scale = 0.01
        
        # Function to convert coordinates
        def to_lat_lon(point):
            return (base_lat + point[1] * scale, base_lon + point[0] * scale)
        
        # Create map centered on warehouse
        warehouse_lat_lon = to_lat_lon(self.warehouse_location)
        m = folium.Map(location=warehouse_lat_lon, zoom_start=12)
        
        # Add warehouse marker
        folium.Marker(
            warehouse_lat_lon,
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
            popup='Warehouse'
        ).add_to(m)
        
        # Add delivery points
        for i, point in enumerate(self.delivery_points):
            lat_lon = to_lat_lon(point)
            demand = self.demands[i+1]  # Skip depot (index 0)
            folium.Marker(
                lat_lon,
                icon=folium.Icon(color='blue', icon='box', prefix='fa'),
                popup=f'Delivery Point {i+1} (Demand: {demand})'
            ).add_to(m)
        
        # Plot routes
        colors = ['green', 'purple', 'orange', 'brown', 'pink']
        
        for i, route_info in enumerate(solution['routes']):
            route = route_info['route']
            color = colors[i % len(colors)]
            
            # Skip empty routes
            if len(route) <= 2:  # Just depot to depot
                continue
            
            # Extract route coordinates
            all_locations = [self.warehouse_location] + self.delivery_points
            route_latlon = [to_lat_lon(all_locations[j]) for j in route]
            
            # Add route line
            folium.PolyLine(
                route_latlon,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Vehicle {i+1} - {route_info['distance']:.2f} km"
            ).add_to(m)
        
        # Add route information as a legend
        route_html = "<h4>Route Summary</h4>"
        route_html += "<table>"
        route_html += "<tr><th>Vehicle</th><th>Distance (km)</th><th>Packages</th><th>CO2 (kg)</th></tr>"
        
        for i, route_info in enumerate(solution['routes']):
            route_html += f"<tr><td>{i+1}</td><td>{route_info['distance']:.2f}</td>"
            route_html += f"<td>{route_info['packages']}</td><td>{route_info['emissions']:.2f}</td></tr>"
        
        route_html += f"<tr><th>Total</th><th>{solution['total_distance']:.2f}</th>"
        route_html += f"<th>{solution['total_packages']}</th><th>{solution['total_emissions']:.2f}</th></tr>"
        route_html += "</table>"
        
        # Add legend to map
        folium.Element(route_html).add_to(m)
        
        if save_file:
            m.save(save_file)
        
        return m
    
    def generate_report(self):
        """Generate a complete report on the route optimization"""
        # Solve the VRP
        solution = self.solve_vrp()
        
        if not solution:
            return "No solution found!"
        
        # Create visualizations
        self.plot_routes(solution, save_file="route_plot.png")
        self.create_map(solution, save_file="route_map.html")
        
        # Create dataframes for reporting
        routes_df = pd.DataFrame(solution['routes'])
        
        # Expand route information
        all_locations = [self.warehouse_location] + self.delivery_points
        location_names = ['Warehouse'] + [f'Delivery {i+1}' for i in range(len(self.delivery_points))]
        
        # Function to convert route indices to location names
        def route_to_names(route):
            return ' → '.join([location_names[i] for i in route])
        
        routes_df['route_names'] = routes_df['route'].apply(route_to_names)
        
        # Print summary
        print("\n=== Route Optimization Results ===")
        print(f"Total Distance: {solution['total_distance']:.2f} km")
        print(f"Total Packages Delivered: {solution['total_packages']}")
        print(f"Total CO2 Emissions: {solution['total_emissions']:.2f} kg")
        print(f"Number of Vehicles Used: {len([r for r in solution['routes'] if len(r['route']) > 2])}")
        
        print("\n=== Vehicle Routes ===")
        for i, route in routes_df.iterrows():
            print(f"Vehicle {route['vehicle_id']+1} ({route['vehicle_type']}): {route['route_names']}")
            print(f"  Distance: {route['distance']:.2f} km | Packages: {route['packages']} | CO2: {route['emissions']:.2f} kg")
        
        # Save to CSV
        routes_df.to_csv("optimized_routes.csv", index=False)
        
        return solution


if __name__ == "__main__":
    # Create optimizer with random delivery points
    optimizer = LogisticsRouteOptimizer(num_delivery_points=15)
    
    # Generate and display report
    solution = optimizer.generate_report()
    
    print("\nResults have been saved to CSV, PNG, and HTML files.")
