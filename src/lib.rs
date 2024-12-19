use numpy::PyArray1;
use pyo3::exceptions::PyIOError;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::io;

use itertools::Itertools;
use maybenot::{event::TriggerEvent, Machine};
use maybenot_simulator::queue::SimQueue;
use maybenot_simulator::{
    network::Network, parse_trace, sim, sim_advanced, SimEvent, SimulatorArgs,
};

use std::{str::FromStr, time::Duration};

fn logs_at_client(trace: &Vec<SimEvent>) -> (Vec<f64>, Vec<i8>, Vec<bool>) {
    let starting_time = trace[0].time;

    let (times, events, paddings): (Vec<f64>, Vec<i8>, Vec<bool>) = trace
        .iter()
        .filter(|p| p.client)
        .map(|p| {
            (
                (p.time - starting_time).as_micros() as f64,
                match p.event {
                    TriggerEvent::TunnelSent => 1,  // Upload -> 1
                    TriggerEvent::TunnelRecv => -1, // Download -> -1
                    _ => 0,
                },
                p.contains_padding,
            )
        })
        .multiunzip();

    (times, events, paddings)
}

fn netwk_monitor(trace: &Vec<SimEvent>) {
    let mut i: usize = 0;
    let n: usize = 1000;
    for p in trace.iter() {
        println!(
            "Time: {:?}, Event: {:?}, Client: {:?}, Padding: {:?}",
            p.time, p.event, p.client, p.contains_padding,
        );
        if i > n {
            break;
        }
        i += 1;
    }
}

fn cast_to_numpy_trace(
    times: Vec<f64>,
    events: Vec<i8>,
    paddings: Vec<bool>,
    py: Python,
) -> (
    Bound<PyArray1<f64>>,
    Bound<PyArray1<i8>>,
    Bound<PyArray1<bool>>,
) {
    let np_times = PyArray1::from_vec(py, times);
    let np_events = PyArray1::from_vec(py, events);
    let np_paddings = PyArray1::from_vec(py, paddings);

    (np_times, np_events, np_paddings)
}

fn convert_machines(machine_strs: Vec<String>) -> Vec<Machine> {
    machine_strs
        .iter()
        .map(|machine_str| Machine::from_str(machine_str).unwrap())
        .collect()
}

fn sim_def_on_trace(
    raw_trace: &str,
    machines_client: Vec<String>,
    machines_server: Vec<String>,
    network_delay_millis: u64,
    max_trace_length: usize,
) -> (Vec<f64>, Vec<i8>, Vec<bool>) {
    let network = Network::new(Duration::from_millis(network_delay_millis), None);
    let machines_client = convert_machines(machines_client);
    let machines_server = convert_machines(machines_server);

    let mut input_trace: SimQueue = parse_trace(&raw_trace, &network);

    let trace: Vec<SimEvent> = sim(
        &machines_client,
        &machines_server,
        &mut input_trace,
        network.delay,
        max_trace_length,
        true,
    );

    let client_logs = logs_at_client(&trace);

    if client_logs.0.len() == 0 {
        netwk_monitor(&trace);
    }

    client_logs
}

fn sim_def_on_trace_advanced(
    raw_trace: &str,
    machines_client: Vec<String>,
    machines_server: Vec<String>,
    network_delay_millis: u64,
    max_trace_length: usize,
    max_padding_frac_client: f64,
    max_padding_frac_server: f64,
    max_blocking_frac_client: f64,
    max_blocking_frac_server: f64,
    debug: bool,
) -> (Vec<f64>, Vec<i8>, Vec<bool>) {
    let network = Network::new(Duration::from_millis(network_delay_millis), None);
    let simulator_args = SimulatorArgs {
        network: &network,
        max_trace_length,
        max_sim_iterations: 0,
        only_client_events: false,
        only_network_activity: !debug,
        max_padding_frac_client,
        max_padding_frac_server,
        max_blocking_frac_client,
        max_blocking_frac_server,
        insecure_rng_seed: None,
        client_integration: None,
        server_integration: None,
    };

    let machines_client = convert_machines(machines_client);
    let machines_server = convert_machines(machines_server);

    let mut input_trace: SimQueue = parse_trace(&raw_trace, &network);
    let input_len = input_trace.len().clone();

    let trace: Vec<SimEvent> = sim_advanced(
        &machines_client,
        &machines_server,
        &mut input_trace,
        &simulator_args,
    );

    let client_logs = logs_at_client(&trace);

    if debug {
        netwk_monitor(&trace);
        println!("Input trace len: {input_len}");
    }
    println!("Simulator args: {:?}", simulator_args);

    client_logs
}

fn load_trace_to_string(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}

///
/// bindings for the maybenot simulator, and some laoding functions.
#[pymodule]
fn rustbindings<'py>(m: Bound<'py, PyModule>) -> PyResult<()> {
    // wrapper of `sim_def`
    #[pyfn(m)]
    #[pyo3(name = "sim_trace")]
    fn sim_trace<'py>(
        py: Python<'py>,
        raw_trace: String,
        machines_client: Vec<String>,
        machines_server: Vec<String>,
        network_delay_millis: u64,
        max_trace_length: usize,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (times, events, paddings) = sim_def_on_trace(
            &raw_trace,
            machines_client,
            machines_server,
            network_delay_millis,
            max_trace_length,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sim_trace_advanced")]
    fn sim_trace_advanced<'py>(
        py: Python<'py>,
        raw_trace: String,
        machines_client: Vec<String>,
        machines_server: Vec<String>,
        network_delay_millis: u64,
        max_trace_length: usize,
        max_padding_frac_client: f64,
        max_padding_frac_server: f64,
        max_blocking_frac_client: f64,
        max_blocking_frac_server: f64,
        debug: bool,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (times, events, paddings) = sim_def_on_trace_advanced(
            &raw_trace,
            machines_client,
            machines_server,
            network_delay_millis,
            max_trace_length,
            max_padding_frac_client,
            max_padding_frac_server,
            max_blocking_frac_client,
            max_blocking_frac_server,
            debug,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sim_trace_from_file")]
    fn sim_trace_from_file<'py>(
        py: Python<'py>,
        path: String,
        machines_client: Vec<String>,
        machines_server: Vec<String>,
        network_delay_millis: u64,
        max_trace_length: usize,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let raw_trace = load_trace_to_string(&path).unwrap();
        let (times, events, paddings) = sim_def_on_trace(
            &raw_trace,
            machines_client,
            machines_server,
            network_delay_millis,
            max_trace_length,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_trace_to_numpy")]
    fn load_trace_to_np<'py>(
        py: Python<'py>,
        path: String,
        network_delay_millis: u64,
        max_trace_length: usize,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let raw_trace = load_trace_to_string(&path).unwrap();

        let (times, events, paddings) = sim_def_on_trace(
            &raw_trace,
            vec![],
            vec![],
            network_delay_millis,
            max_trace_length,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_trace_to_str")]
    fn load_trace_to_str(path: String) -> PyResult<String> {
        match load_trace_to_string(&path) {
            Ok(content) => Ok(content), // Return the file content
            Err(e) => Err(PyIOError::new_err(format!("Failed to read file: {}", e))),
        }
    }

    Ok(())
}
