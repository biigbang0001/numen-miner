use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use serde::Serialize;
use std::fmt;
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a GPU has to stay at `--temp-stop` before it pauses.
const SUSTAIN: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Default)]
pub struct Reading {
    pub temp: Option<f64>,
    pub fan: Option<Fan>,
    pub power: Option<f64>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fan {
    Percent(u32),
    Rpm(u64),
}

impl fmt::Display for Fan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fan::Percent(percent) => write!(f, "{percent}%"),
            Fan::Rpm(rpm) => write!(f, "{rpm} rpm"),
        }
    }
}

pub enum Sensor {
    Nvml { nvml: Arc<Nvml>, bus: String },
    #[cfg(target_os = "linux")]
    Hwmon(PathBuf),
    #[cfg(windows)]
    Adlx { adlx: Arc<Adlx>, index: usize },
}

impl Sensor {
    pub fn read(&self) -> Reading {
        match self {
            Sensor::Nvml { nvml, bus } => {
                let Ok(device) = nvml.device_by_pci_bus_id(bus.as_str()) else {
                    return Reading::default();
                };
                Reading {
                    temp: device.temperature(TemperatureSensor::Gpu).ok().map(f64::from),
                    fan: device.fan_speed(0).ok().map(Fan::Percent),
                    power: device.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
                }
            }
            #[cfg(target_os = "linux")]
            Sensor::Hwmon(dir) => read_hwmon(dir),
            #[cfg(windows)]
            Sensor::Adlx { adlx, index } => adlx.read(*index),
        }
    }
}

/// Where AMD readings come from, the gfx driver's sysfs on Linux and ADLX
/// on Windows.
pub struct AmdSensors {
    #[cfg(windows)]
    adlx: Option<Arc<Adlx>>,
}

impl AmdSensors {
    pub fn new() -> Self {
        AmdSensors {
            #[cfg(windows)]
            adlx: Adlx::new().map(Arc::new),
        }
    }

    pub fn sensor(&self, pci: [i32; 3]) -> Option<Sensor> {
        #[cfg(target_os = "linux")]
        {
            hwmon(pci).map(Sensor::Hwmon)
        }
        #[cfg(windows)]
        {
            let adlx = self.adlx.as_ref()?;
            let index = adlx.find(pci)?;
            Some(Sensor::Adlx { adlx: adlx.clone(), index })
        }
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            let _ = pci;
            None
        }
    }
}

/// The gfx driver's hwmon directory for the GPU at this PCI location.
#[cfg(target_os = "linux")]
fn hwmon(pci: [i32; 3]) -> Option<PathBuf> {
    let device = format!("/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.0/hwmon", pci[0], pci[1], pci[2]);
    std::fs::read_dir(device).ok()?.flatten().map(|entry| entry.path()).next()
}

#[cfg(target_os = "linux")]
fn read_hwmon(dir: &Path) -> Reading {
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok()?.trim().parse::<u64>().ok();
    Reading {
        temp: read("temp1_input").map(|millidegrees| millidegrees as f64 / 1000.0),
        fan: read("fan1_input").map(Fan::Rpm),
        power: read("power1_average").or_else(|| read("power1_input")).map(|microwatts| microwatts as f64 / 1e6),
    }
}

/// ADLX with the PCI location of every GPU it lists.
#[cfg(windows)]
pub struct Adlx {
    gpus: Vec<(u32, u32, adlx::Gpu)>,
    services: adlx::PerformanceMonitoringServices,
    _helper: adlx::AdlxHelper,
}

#[cfg(windows)]
impl Adlx {
    fn new() -> Option<Self> {
        let helper = adlx::AdlxHelper::new().ok()?;
        let system = helper.system();
        let services = system.performance_monitoring_services().ok()?;
        let list = system.gpus().ok()?;
        let gpus = (0..list.size())
            .filter_map(|i| {
                let gpu = list.at(i).ok()?;
                let (bus, device) = pci_location(gpu.pnp_string().ok()?)?;
                Some((bus, device, gpu))
            })
            .collect();
        Some(Adlx { gpus, services, _helper: helper })
    }

    fn find(&self, pci: [i32; 3]) -> Option<usize> {
        let (bus, device) = (pci[1] as u32, pci[2] as u32);
        self.gpus.iter().position(|(b, d, _)| (*b, *d) == (bus, device))
    }

    fn read(&self, index: usize) -> Reading {
        let Ok(metrics) = self.services.current_gpu_metrics(&self.gpus[index].2) else {
            return Reading::default();
        };
        Reading {
            temp: metrics.temperature().ok(),
            fan: metrics.fan_speed().ok().map(|rpm| Fan::Rpm(rpm as u64)),
            power: metrics.power().ok(),
        }
    }
}

/// PCI bus and device numbers of the device with this instance id, read
/// through the configuration manager.
#[cfg(windows)]
fn pci_location(instance_id: &str) -> Option<(u32, u32)> {
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_Get_DevNode_PropertyW, CM_LOCATE_DEVNODE_NORMAL, CM_Locate_DevNodeW, CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::Properties::{DEVPKEY_Device_Address, DEVPKEY_Device_BusNumber};
    use windows_sys::Win32::Foundation::DEVPROPKEY;

    let id: Vec<u16> = instance_id.encode_utf16().chain(Some(0)).collect();
    let mut node = 0;
    if unsafe { CM_Locate_DevNodeW(&mut node, id.as_ptr(), CM_LOCATE_DEVNODE_NORMAL) } != CR_SUCCESS {
        return None;
    }
    let property = |key: &DEVPROPKEY| {
        let mut kind = 0;
        let mut value = 0u32;
        let mut size = size_of::<u32>() as u32;
        let status = unsafe {
            CM_Get_DevNode_PropertyW(node, key, &mut kind, (&raw mut value).cast(), &mut size, 0)
        };
        (status == CR_SUCCESS).then_some(value)
    };
    let bus = property(&DEVPKEY_Device_BusNumber)?;
    let address = property(&DEVPKEY_Device_Address)?;
    Some((bus, address >> 16))
}

#[derive(Clone, Copy)]
pub struct TempLimit {
    pub stop: f64,
    pub start: f64,
}

/// Whether a GPU should be paused after this reading, and since when it has
/// run hot. A GPU pauses once it stays at the stop temperature for
/// [`SUSTAIN`] and resumes at the start temperature.
pub fn throttle(
    paused: bool,
    hot_since: Option<Instant>,
    temp: f64,
    limit: TempLimit,
    now: Instant,
) -> (bool, Option<Instant>) {
    if paused {
        return (temp > limit.start, None);
    }
    if temp < limit.stop {
        return (false, None);
    }
    let since = hot_since.unwrap_or(now);
    (now.duration_since(since) >= SUSTAIN, Some(since))
}
