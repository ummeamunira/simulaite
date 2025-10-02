import streamlit as st
import simpy
import random
import pandas as pd
import graphviz
import time # Used for unique simulation ID

# --- CONFIGURATION ---
SIM_DURATION = 480  # Simulation time in minutes (e.g., one 8-hour shift)

# --- SIMULATION LOGIC (SimPy) ---

def factory_process(env, name, raw_material_name, machine_resources, processing_times, metrics_data, machine_queue_wip, wip_tracking_dict):
    """
    Models the journey of a single item through the factory.
    """
    arrival_time = env.now
    
    # Initialize a list to hold the machine start/end times for this item
    machine_process_trace = []
    
    # Add item to WIP tracking dictionary
    wip_tracking_dict[name] = {
        'arrival_time': arrival_time,
        'current_status': f'Waiting for {list(machine_resources.keys())[0]}',
        'machine_trace': []
    }

    # Loop through the user-defined machines in order
    machine_names = list(machine_resources.keys())
    
    for i, machine_name in enumerate(machine_names):
        capacity_resource = machine_resources[machine_name]
        process_time = processing_times[machine_name]
        
        # 1. Request the machine (Item enters the queue/WIP)
        with capacity_resource.request() as req:
            
            # Update WIP tracking status
            wip_tracking_dict[name]['current_status'] = f'In Queue for {machine_name}'
            machine_queue_wip[machine_name] += 1
            
            yield req
            
            # --- START TIME RECORDED HERE ---
            start_time = env.now 
            
            # Item is now being processed (leaving the queue)
            machine_queue_wip[machine_name] -= 1
            wip_tracking_dict[name]['current_status'] = f'Processing on {machine_name}'

            # 2. Process the item
            yield env.timeout(process_time)
            
            # --- END TIME RECORDED HERE ---
            end_time = env.now
            
            # Record the start and end time for this machine
            machine_process_trace.append({
                'machine': machine_name,
                'start_time': start_time,
                'end_time': end_time
            })
            
        # 3. Release the machine (Item moves to the next stage)

    # Item is finished
    departure_time = env.now
    cycle_time = departure_time - arrival_time
    
    # Remove item from WIP tracking dictionary
    del wip_tracking_dict[name]
    
    # Record metrics for finished product
    metrics_data.append({
        'name': name,
        'rm_used': raw_material_name,
        'arrival_time': arrival_time,
        'departure_time': departure_time,
        'cycle_time': cycle_time,
        'machine_trace': machine_process_trace # NEW: Full trace data
    })


# --- raw_material_generator ---
def raw_material_generator(env, machine_resources, processing_times, metrics_data, rm_name, arrival_rate, machine_queue_wip, wip_tracking_dict):
    """
    Generates raw materials based on the defined arrival rate.
    """
    item_counter = 0
    # Inter-arrival time is 60 minutes / arrival_rate
    inter_arrival_time = 60 / arrival_rate if arrival_rate > 0 else SIM_DURATION

    while True:
        # Generate item and start its process flow
        item_counter += 1
        
        # FIX: Remove the slicing [:2] to ensure the full rm_name is used in the unique ID
        item_name = f"Product-{env.now:0.2f}-{rm_name}-{item_counter}" 
        
        env.process(factory_process(env, item_name, rm_name, machine_resources, processing_times, metrics_data, machine_queue_wip, wip_tracking_dict))
        
        # Wait for the next arrival
        yield env.timeout(random.expovariate(1.0/inter_arrival_time))


def run_simulation(config):
    """
    Initializes and runs the SimPy environment.
    """
    # 1. Initialize environment and metrics
    env = simpy.Environment()
    metrics_data = []
    
    # NEW: Dictionary to track all WIP items by name
    wip_tracking_dict = {} 
    
    # Initialize a dictionary to track WIP in queues per machine
    machine_queue_wip = {name: 0 for name in config['machines'].keys()}
    
    # 2. Create Machine Resources
    machine_resources = {}
    processing_times = {}
    
    for i, (name, time_val) in enumerate(config['machines'].items()):
        machine_resources[name] = simpy.Resource(env, capacity=1)
        processing_times[name] = time_val
    
    # 3. Start Raw Material Generators
    for rm_name, rate in config['raw_materials'].items():
        env.process(raw_material_generator(env, machine_resources, processing_times, metrics_data, rm_name, rate, machine_queue_wip, wip_tracking_dict))

    # 4. Run the simulation
    env.run(until=SIM_DURATION)
    
    # 5. Process and return results
    df = pd.DataFrame(metrics_data)
    
    # Ending WIP is the sum of items remaining in all machine queues + items being processed
    final_wip_value = len(wip_tracking_dict)
    
    if df.empty:
        return None, "No products completed during the simulation run.", final_wip_value, wip_tracking_dict
    
    # Calculate key metrics
    total_products = len(df)
    avg_cycle_time = df['cycle_time'].mean()
    throughput = total_products / (SIM_DURATION / 60) # Products per hour
    
    return {
        'total_products': total_products,
        'avg_cycle_time': avg_cycle_time,
        'throughput': throughput,
        'final_wip': final_wip_value,
        'data_frame': df,
        'machine_queue_wip': machine_queue_wip 
    }, None, final_wip_value, wip_tracking_dict


# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🏭 Factory Flow Simulation Demo")
st.markdown("Configure your factory production line, run the simulation, and analyze the key operational metrics.")

# --- USER INPUT SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Factory Configuration")
    
    # --- Product and Raw Materials ---
    st.subheader("Product & Raw Materials")
    final_product_name = st.text_input("Final Product Name", "Custom Widget")
    
    num_rms = st.number_input("Number of Raw Materials", 1, 4, 2)
    raw_materials = {}
    for i in range(1, num_rms + 1):
        col1, col2 = st.columns(2)
        rm_name = col1.text_input(f"RM {i} Name", f"RM{i}")
        rm_rate = col2.number_input(f"RM {i} Arrival Rate (items/hr)", 1, 100, 10)
        raw_materials[rm_name] = rm_rate
    
    # --- Machines ---
    st.subheader("Machines & Process Time (minutes)")
    
    num_machines = st.number_input("Number of Machines", 1, 5, 3)
    machines = {}
    for i in range(1, num_machines + 1):
        col1, col2 = st.columns(2)
        machine_name = col1.text_input(f"Machine {i} Name", f"Machine_{chr(65+i-1)}")
        process_time = col2.number_input(f"Time (min) on {machine_name}", 1, 60, 10) 
        machines[machine_name] = process_time
        
    st.info(f"**Simulation Run Time:** {SIM_DURATION // 60} hours ({SIM_DURATION} minutes)")

# --- RUN SIMULATION BUTTON ---
if st.button("🚀 Run Simulation"):
    
    sim_config = {
        'raw_materials': raw_materials,
        'machines': machines
    }
    
    # Run the simulation and unpack the four returned values
    with st.spinner('Running discrete-event simulation...'):
        results, error, final_wip_value, wip_tracking_dict = run_simulation(sim_config)

    if error:
        st.error(error)
        st.metric("Ending WIP (items)", f"{final_wip_value}")
    elif results:
        st.subheader(f"✅ Simulation Complete: Metrics for {final_product_name}")
        
        # --- 1. Display Key Metrics ---
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_arrival_rate = sum(raw_materials.values())
        
        col1.metric("Total Arrival Rate (items/hr)", f"{total_arrival_rate:.1f}")
        col2.metric("Total Products Produced", f"{results['total_products']}") 
        col3.metric("Average Cycle Time (min)", f"{results['avg_cycle_time']:.2f}")
        col4.metric("Throughput (products/hr)", f"{results['throughput']:.2f}")
        col5.metric("Ending WIP (items)", f"{results['final_wip']}")
        
        st.markdown("---")

        # --- 2. Process Flow Diagram (Graphviz) with WIP Count ---
        st.subheader("Process Flow Diagram (WIP in Queue)")
        
        graph = graphviz.Digraph(comment='Factory Flow', graph_attr={'rankdir': 'LR'})
        
        graph.node("START", "Raw Materials Arrive", shape='box', style='filled', color='lightgreen')
        
        machine_nodes = list(machines.keys())
        for node in machine_nodes:
            wip_count = results['machine_queue_wip'][node]
            time_val = machines[node]
            graph.node(node, f"{node}\n({time_val} min)\n[WIP Queue: {wip_count}]", 
                        shape='cylinder', style='filled', color='lightblue')
            
        graph.node("END", final_product_name, shape='diamond', style='filled', color='gold')
        
        if machine_nodes:
            first_machine = machine_nodes[0]
            for rm_name in raw_materials.keys():
                 graph.edge("START", first_machine, label=f"RM: {rm_name}")

            for i in range(len(machine_nodes) - 1):
                graph.edge(machine_nodes[i], machine_nodes[i+1], label="WIP")
                
            graph.edge(machine_nodes[-1], "END", label="Final Product")
        
        st.graphviz_chart(graph)
        
        st.markdown("---")
        
        # --- 3. WIP Data Table ---
        st.subheader("🚧 Ending Work In Process (WIP) Details")
        
        # Prepare data for WIP table
        wip_data_list = []
        for name, data in wip_tracking_dict.items():
            wip_data_list.append({
                'Product Name': name,
                'Status': data['current_status'],
                'Time In System (min)': f"{(SIM_DURATION - data['arrival_time']):.2f}"
            })

        wip_df = pd.DataFrame(wip_data_list)
        if not wip_df.empty:
            st.dataframe(wip_df, use_container_width=True)
        else:
            st.success("🎉 No items were left in process (WIP) at the end of the simulation!")


        # --- 4. Finished Products Raw Data (with Machine Trace) ---
        st.subheader("✅ Finished Products Raw Data")
        
        # Flatten the machine trace data into a new, easily viewable DataFrame
        finished_products_list = []
        for _, row in results['data_frame'].iterrows():
            item_data = {
                'Product Name': row['name'],
                'RM Used': row['rm_used'],
                'Arrival Time (min)': f"{row['arrival_time']:.2f}",
                'Departure Time (min)': f"{row['departure_time']:.2f}",
                'Cycle Time (min)': f"{row['cycle_time']:.2f}"
            }
            
            # Add machine start/end times as new columns
            for trace in row['machine_trace']:
                item_data[f"Start {trace['machine']} (min)"] = f"{trace['start_time']:.2f}"
                item_data[f"End {trace['machine']} (min)"] = f"{trace['end_time']:.2f}"
                
            finished_products_list.append(item_data)
            
        finished_df = pd.DataFrame(finished_products_list)
        
        with st.expander("Expand to view detailed trace data"):
            st.dataframe(finished_df, use_container_width=True)
            
        st.markdown("---")
        
        # --- 5. Little's Law Check (Conceptual) ---
        st.subheader("Little's Law Check (Conceptual)")
        st.info(
            f"**WIP** (Total items still in process) at end: **{results['final_wip']}** items. \n"
            f"*(In a steady-state system: **WIP = Throughput × Cycle Time**)*"
        )