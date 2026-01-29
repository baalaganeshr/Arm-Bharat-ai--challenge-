"""
FPGA Interface Module
Handles UART communication between Raspberry Pi/Host and FPGA

Features:
- Serial UART communication at configurable baud rate
- Image data serialization for FPGA processing
- Result reception and parsing
- Error handling and timeouts
- Connection management
"""

import serial
import serial.tools.list_ports
import numpy as np
import time
import struct
from typing import Optional, Tuple, List
from enum import Enum


class FPGACommand(Enum):
    """FPGA communication commands"""
    START_FRAME = 0xAA
    END_FRAME = 0x55
    ACK = 0x06
    NAK = 0x15
    PROCESS_IMAGE = 0x01
    GET_STATUS = 0x02
    RESET = 0x03


class FPGAInterface:
    """
    Interface for communicating with FPGA via UART
    
    Attributes:
        port: Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
        baudrate: Communication speed (default 115200)
        timeout: Read timeout in seconds
    """
    
    def __init__(self, port: str = None, baudrate: int = 115200, timeout: float = 2.0):
        """
        Initialize UART connection to FPGA
        
        Args:
            port: Serial port (auto-detect if None)
            baudrate: Baud rate (default 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.img_size = 64  # Expected image size
        
        # Statistics
        self.frames_sent = 0
        self.frames_received = 0
        self.errors = 0
        self.total_latency = 0.0
        
        # Auto-detect port if not specified
        if self.port is None:
            self.port = self._auto_detect_port()
        
        # Attempt connection
        if self.port:
            self._connect()
    
    def _auto_detect_port(self) -> Optional[str]:
        """Auto-detect available serial ports"""
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("[WARN] No serial ports found")
            return None
        
        print("[INFO] Available serial ports:")
        for p in ports:
            print(f"       {p.device} - {p.description}")
        
        # Return first available port
        return ports[0].device
    
    def _connect(self) -> bool:
        """Establish serial connection"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            # Flush buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            self.connected = True
            print(f"[INFO] Connected to FPGA on {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"[ERROR] Could not connect to FPGA: {e}")
            self.ser = None
            self.connected = False
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to FPGA"""
        self.close()
        time.sleep(1)
        return self._connect()
    
    def is_connected(self) -> bool:
        """Check if connected to FPGA"""
        return self.connected and self.ser is not None and self.ser.is_open
    
    def send_image(self, image_data: np.ndarray) -> bool:
        """
        Send image data to FPGA for processing
        
        Args:
            image_data: Numpy array of shape (64, 64) with values 0-255 or 0.0-1.0
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected():
            print("[ERROR] Not connected to FPGA")
            return False
        
        try:
            # Validate shape
            if image_data.shape != (self.img_size, self.img_size):
                print(f"[ERROR] Image must be {self.img_size}x{self.img_size}")
                return False
            
            # Normalize to uint8 if needed
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                image_data = (image_data * 255).astype(np.uint8)
            elif image_data.dtype != np.uint8:
                image_data = image_data.astype(np.uint8)
            
            # Prepare packet
            # Header: START_FRAME + PROCESS_IMAGE + SIZE_HIGH + SIZE_LOW
            data_size = self.img_size * self.img_size
            header = bytes([
                FPGACommand.START_FRAME.value,
                FPGACommand.PROCESS_IMAGE.value,
                (data_size >> 8) & 0xFF,
                data_size & 0xFF
            ])
            
            # Image data (flattened)
            image_bytes = image_data.flatten().tobytes()
            
            # Checksum (simple XOR)
            checksum = 0
            for b in image_bytes:
                checksum ^= b
            
            # Footer: CHECKSUM + END_FRAME
            footer = bytes([checksum, FPGACommand.END_FRAME.value])
            
            # Send packet
            packet = header + image_bytes + footer
            self.ser.write(packet)
            self.ser.flush()
            
            self.frames_sent += 1
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to send image: {e}")
            self.errors += 1
            return False
    
    def receive_result(self) -> Optional[Tuple[int, float]]:
        """
        Receive processed result from FPGA
        
        Returns:
            Tuple of (class_id, confidence) or None on error
            class_id: 0 = mask, 1 = no_mask
        """
        if not self.is_connected():
            return None
        
        try:
            # Read response header (4 bytes)
            header = self.ser.read(4)
            
            if len(header) < 4:
                print("[WARN] Timeout waiting for FPGA response")
                return None
            
            # Parse header
            if header[0] != FPGACommand.START_FRAME.value:
                print(f"[WARN] Invalid response header: {header.hex()}")
                return None
            
            # Read result data (4 bytes: class + confidence float)
            data = self.ser.read(5)  # class(1) + confidence(4 bytes float)
            
            if len(data) < 5:
                print("[WARN] Incomplete response data")
                return None
            
            # Parse result
            class_id = data[0]
            confidence = struct.unpack('<f', data[1:5])[0]
            
            # Read footer
            footer = self.ser.read(1)
            if len(footer) > 0 and footer[0] == FPGACommand.END_FRAME.value:
                self.frames_received += 1
                return (class_id, confidence)
            
            return (class_id, confidence)
            
        except Exception as e:
            print(f"[ERROR] Failed to receive result: {e}")
            self.errors += 1
            return None
    
    def process_image(self, image_data: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Send image and receive result in one call
        
        Args:
            image_data: 64x64 grayscale image
            
        Returns:
            Tuple of (class_id, confidence) or None on error
        """
        start_time = time.time()
        
        if self.send_image(image_data):
            result = self.receive_result()
            
            if result is not None:
                latency = time.time() - start_time
                self.total_latency += latency
                return result
        
        return None
    
    def get_status(self) -> Optional[dict]:
        """Get FPGA status information"""
        if not self.is_connected():
            return None
        
        try:
            # Send status request
            status_cmd = bytes([
                FPGACommand.START_FRAME.value,
                FPGACommand.GET_STATUS.value,
                FPGACommand.END_FRAME.value
            ])
            self.ser.write(status_cmd)
            self.ser.flush()
            
            # Read response
            response = self.ser.read(16)
            
            if len(response) >= 8:
                return {
                    'version': response[2],
                    'temperature': response[3],
                    'utilization': response[4],
                    'ready': response[5] == 1
                }
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to get status: {e}")
            return None
    
    def reset(self) -> bool:
        """Reset FPGA accelerator"""
        if not self.is_connected():
            return False
        
        try:
            reset_cmd = bytes([
                FPGACommand.START_FRAME.value,
                FPGACommand.RESET.value,
                FPGACommand.END_FRAME.value
            ])
            self.ser.write(reset_cmd)
            self.ser.flush()
            
            time.sleep(0.5)
            
            # Wait for ACK
            response = self.ser.read(1)
            return len(response) > 0 and response[0] == FPGACommand.ACK.value
            
        except Exception as e:
            print(f"[ERROR] Failed to reset FPGA: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get communication statistics"""
        avg_latency = 0
        if self.frames_received > 0:
            avg_latency = self.total_latency / self.frames_received * 1000  # ms
        
        return {
            'frames_sent': self.frames_sent,
            'frames_received': self.frames_received,
            'errors': self.errors,
            'avg_latency_ms': avg_latency,
            'connected': self.is_connected()
        }
    
    def close(self):
        """Close serial connection"""
        if self.ser is not None:
            try:
                self.ser.close()
            except:
                pass
            self.ser = None
        
        self.connected = False
        print("[INFO] FPGA connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


class FPGASimulator:
    """
    Simulated FPGA for testing without hardware
    Mimics FPGA behavior for development/testing
    """
    
    def __init__(self, latency_ms: float = 5.0):
        """
        Initialize simulator
        
        Args:
            latency_ms: Simulated processing latency in milliseconds
        """
        self.latency_ms = latency_ms
        self.connected = True
        self.frames_processed = 0
    
    def is_connected(self) -> bool:
        return self.connected
    
    def process_image(self, image_data: np.ndarray) -> Tuple[int, float]:
        """
        Simulate FPGA processing
        Returns random classification for testing
        """
        # Simulate processing time
        time.sleep(self.latency_ms / 1000.0)
        
        # Simple threshold-based "classification" for testing
        avg_intensity = np.mean(image_data)
        
        # Simulate classification based on image intensity
        # (Real FPGA would run actual CNN inference)
        if avg_intensity > 0.5:
            class_id = 0  # mask
            confidence = 0.7 + np.random.random() * 0.25
        else:
            class_id = 1  # no_mask
            confidence = 0.7 + np.random.random() * 0.25
        
        self.frames_processed += 1
        return (class_id, confidence)
    
    def get_statistics(self) -> dict:
        return {
            'frames_processed': self.frames_processed,
            'simulated_latency_ms': self.latency_ms,
            'connected': True,
            'mode': 'simulator'
        }
    
    def close(self):
        self.connected = False


def list_available_ports() -> List[str]:
    """List all available serial ports"""
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]


# ============== TESTING ==============
if __name__ == "__main__":
    print("=" * 50)
    print("FPGA Interface Test")
    print("=" * 50)
    
    # List available ports
    ports = list_available_ports()
    print(f"\nAvailable ports: {ports}")
    
    # Test with simulator
    print("\n[TEST] Using FPGA Simulator...")
    sim = FPGASimulator(latency_ms=5.0)
    
    # Create test image
    test_image = np.random.rand(64, 64).astype(np.float32)
    
    # Process multiple frames
    for i in range(5):
        start = time.time()
        result = sim.process_image(test_image)
        elapsed = (time.time() - start) * 1000
        
        class_name = "Mask" if result[0] == 0 else "No Mask"
        print(f"  Frame {i+1}: {class_name} ({result[1]*100:.1f}%) - {elapsed:.1f}ms")
    
    print(f"\nSimulator stats: {sim.get_statistics()}")
    
    # Test real FPGA if port available
    if ports:
        print(f"\n[TEST] Attempting real FPGA connection on {ports[0]}...")
        try:
            with FPGAInterface(port=ports[0], timeout=1.0) as fpga:
                if fpga.is_connected():
                    result = fpga.process_image(test_image)
                    if result:
                        print(f"  Result: Class={result[0]}, Confidence={result[1]:.2f}")
                    print(f"  Stats: {fpga.get_statistics()}")
                else:
                    print("  Could not connect to FPGA")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n[TEST] Complete!")
