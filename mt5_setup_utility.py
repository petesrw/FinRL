#!/usr/bin/env python3
"""
MT5 Setup Utility - Cross-Platform MT5 Path Detection and Configuration
Supports Windows, macOS (Wine/CrossOver), and Linux (Wine)
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

class MT5PathDetector:
    """Cross-platform MT5 path detection utility"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the utility"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def detect_all_mt5_installations(self) -> Dict[str, List[str]]:
        """Detect all MT5 installations on the system"""
        installations = {
            'found_paths': [],
            'data_paths': [],
            'wine_prefixes': [],
            'recommendations': []
        }
        
        if self.system == 'windows':
            installations.update(self._detect_windows_installations())
        elif self.system == 'darwin':  # macOS
            installations.update(self._detect_macos_installations())
        elif self.system == 'linux':
            installations.update(self._detect_linux_installations())
        
        return installations
    
    def _detect_windows_installations(self) -> Dict[str, List[str]]:
        """Detect MT5 installations on Windows"""
        found_paths = []
        data_paths = []
        recommendations = []
        
        # Common Windows installation paths
        search_paths = [
            r'C:\Program Files\MetaTrader 5\terminal64.exe',
            r'C:\Program Files (x86)\MetaTrader 5\terminal64.exe',
            r'%LOCALAPPDATA%\Programs\MetaTrader 5\terminal64.exe',
            r'%PROGRAMFILES%\MetaTrader 5\terminal64.exe',
            r'%PROGRAMFILES(X86)%\MetaTrader 5\terminal64.exe',
            r'D:\MetaTrader 5\terminal64.exe',
            r'E:\MetaTrader 5\terminal64.exe',
            r'C:\MT5\terminal64.exe',
            r'D:\MT5\terminal64.exe'
        ]
        
        # Check each path
        for path in search_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                found_paths.append(expanded_path)
                self.logger.info(f"Found MT5 installation: {expanded_path}")
        
        # Common data paths
        data_search_paths = [
            r'%APPDATA%\MetaQuotes\Terminal',
            r'%LOCALAPPDATA%\MetaQuotes\Terminal',
            r'C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal'
        ]
        
        for path in data_search_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                data_paths.append(expanded_path)
        
        # Registry search (Windows specific)
        try:
            registry_paths = self._search_windows_registry()
            found_paths.extend(registry_paths)
        except Exception as e:
            self.logger.warning(f"Registry search failed: {e}")
        
        # Recommendations
        if not found_paths:
            recommendations.extend([
                "Download MT5 from: https://www.metatrader5.com/en/download",
                "Install to default location: C:\\Program Files\\MetaTrader 5\\",
                "Enable automated trading in MT5: Tools → Options → Expert Advisors"
            ])
        
        return {
            'found_paths': found_paths,
            'data_paths': data_paths,
            'wine_prefixes': [],
            'recommendations': recommendations
        }
    
    def _detect_macos_installations(self) -> Dict[str, List[str]]:
        """Detect MT5 installations on macOS (Wine/CrossOver)"""
        found_paths = []
        data_paths = []
        wine_prefixes = []
        recommendations = []
        
        # Wine installation paths
        wine_paths = [
            '~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe',
            '~/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe',
            '~/.wine/drive_c/MT5/terminal64.exe'
        ]
        
        # CrossOver installation paths
        crossover_paths = [
            '~/Library/Application Support/CrossOver/Bottles/*/drive_c/Program Files/MetaTrader 5/terminal64.exe',
            '~/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine'
        ]
        
        # Check Wine paths
        for path in wine_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                found_paths.append(expanded_path)
                wine_prefixes.append(os.path.dirname(os.path.dirname(os.path.dirname(expanded_path))))
        
        # Check CrossOver paths
        import glob
        for pattern in crossover_paths:
            expanded_pattern = os.path.expanduser(pattern)
            matches = glob.glob(expanded_pattern)
            found_paths.extend(matches)
        
        # Data paths
        data_search_paths = [
            '~/.wine/drive_c/users/*/Application Data/MetaQuotes/Terminal',
            '~/.wine/drive_c/users/crossover/Application Data/MetaQuotes/Terminal',
            '~/Library/Application Support/CrossOver/Bottles/*/drive_c/users/*/Application Data/MetaQuotes/Terminal'
        ]
        
        for pattern in data_search_paths:
            expanded_pattern = os.path.expanduser(pattern)
            matches = glob.glob(expanded_pattern)
            data_paths.extend(matches)
        
        # Check if Wine is installed
        wine_installed = self._check_wine_installation()
        crossover_installed = self._check_crossover_installation()
        
        # Recommendations
        if not found_paths:
            if not wine_installed and not crossover_installed:
                recommendations.extend([
                    "Install Wine: brew install wine-stable",
                    "Or install CrossOver: https://www.codeweavers.com/crossover",
                    "Then install MT5 through Wine/CrossOver"
                ])
            else:
                recommendations.extend([
                    "Download MT5 Windows version",
                    "Install through Wine: wine MetaTrader5Setup.exe",
                    "Or use CrossOver to install MT5"
                ])
        
        return {
            'found_paths': found_paths,
            'data_paths': data_paths,
            'wine_prefixes': wine_prefixes,
            'recommendations': recommendations
        }
    
    def _detect_linux_installations(self) -> Dict[str, List[str]]:
        """Detect MT5 installations on Linux (Wine)"""
        found_paths = []
        data_paths = []
        wine_prefixes = []
        recommendations = []
        
        # Wine installation paths
        wine_paths = [
            '~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe',
            '~/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe',
            '~/.wine/drive_c/MT5/terminal64.exe'
        ]
        
        # PlayOnLinux paths
        pol_paths = [
            '~/.PlayOnLinux/wineprefix/*/drive_c/Program Files/MetaTrader 5/terminal64.exe',
            '~/.PlayOnLinux/wineprefix/MetaTrader5/drive_c/Program Files/MetaTrader 5/terminal64.exe'
        ]
        
        # Lutris paths
        lutris_paths = [
            '~/Games/*/drive_c/Program Files/MetaTrader 5/terminal64.exe'
        ]
        
        # Check all paths
        import glob
        all_patterns = wine_paths + pol_paths + lutris_paths
        
        for pattern in all_patterns:
            expanded_pattern = os.path.expanduser(pattern)
            if '*' in expanded_pattern:
                matches = glob.glob(expanded_pattern)
                found_paths.extend(matches)
            else:
                if os.path.exists(expanded_pattern):
                    found_paths.append(expanded_pattern)
        
        # Find Wine prefixes
        for path in found_paths:
            if 'drive_c' in path:
                prefix_path = path.split('drive_c')[0].rstrip('/')
                wine_prefixes.append(prefix_path)
        
        # Data paths
        data_patterns = [
            '~/.wine/drive_c/users/*/Application Data/MetaQuotes/Terminal',
            '~/.PlayOnLinux/wineprefix/*/drive_c/users/*/Application Data/MetaQuotes/Terminal'
        ]
        
        for pattern in data_patterns:
            expanded_pattern = os.path.expanduser(pattern)
            matches = glob.glob(expanded_pattern)
            data_paths.extend(matches)
        
        # Check Wine installation
        wine_installed = self._check_wine_installation()
        
        # Recommendations
        if not found_paths:
            if not wine_installed:
                recommendations.extend([
                    "Install Wine: sudo apt install wine (Ubuntu/Debian)",
                    "Or: sudo dnf install wine (Fedora)",
                    "Or: sudo pacman -S wine (Arch)",
                    "Then install MT5: wine MetaTrader5Setup.exe"
                ])
            else:
                recommendations.extend([
                    "Download MT5 Windows version",
                    "Install through Wine: wine MetaTrader5Setup.exe",
                    "Configure Wine prefix if needed"
                ])
        
        return {
            'found_paths': found_paths,
            'data_paths': data_paths,
            'wine_prefixes': wine_prefixes,
            'recommendations': recommendations
        }
    
    def _search_windows_registry(self) -> List[str]:
        """Search Windows registry for MT5 installations"""
        found_paths = []
        
        try:
            import winreg
            
            # Registry keys to check
            registry_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\MetaQuotes\MetaTrader 5"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\MetaQuotes\MetaTrader 5"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\MetaQuotes\MetaTrader 5")
            ]
            
            for hkey, subkey in registry_keys:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                        terminal_path = os.path.join(install_path, "terminal64.exe")
                        if os.path.exists(terminal_path):
                            found_paths.append(terminal_path)
                except (FileNotFoundError, OSError):
                    continue
                    
        except ImportError:
            # winreg not available (not Windows)
            pass
        
        return found_paths
    
    def _check_wine_installation(self) -> bool:
        """Check if Wine is installed"""
        try:
            result = subprocess.run(['wine', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_crossover_installation(self) -> bool:
        """Check if CrossOver is installed (macOS)"""
        crossover_paths = [
            '/Applications/CrossOver.app',
            '~/Applications/CrossOver.app'
        ]
        
        for path in crossover_paths:
            if os.path.exists(os.path.expanduser(path)):
                return True
        return False
    
    def test_mt5_connection(self, mt5_path: str) -> Tuple[bool, str]:
        """Test if MT5 can be launched and connected"""
        try:
            if self.system == 'windows':
                return self._test_windows_mt5(mt5_path)
            elif self.system in ['darwin', 'linux']:
                return self._test_wine_mt5(mt5_path)
        except Exception as e:
            return False, f"Test failed: {str(e)}"
        
        return False, "Unsupported system"
    
    def _test_windows_mt5(self, mt5_path: str) -> Tuple[bool, str]:
        """Test MT5 on Windows"""
        if not os.path.exists(mt5_path):
            return False, f"MT5 executable not found: {mt5_path}"
        
        try:
            # Try to import MetaTrader5 module
            import MetaTrader5 as mt5
            
            # Try to initialize
            if mt5.initialize(path=mt5_path):
                mt5.shutdown()
                return True, "MT5 connection successful"
            else:
                error = mt5.last_error()
                return False, f"MT5 initialization failed: {error}"
                
        except ImportError:
            return False, "MetaTrader5 Python package not installed"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def _test_wine_mt5(self, mt5_path: str) -> Tuple[bool, str]:
        """Test MT5 through Wine"""
        if not os.path.exists(mt5_path):
            return False, f"MT5 executable not found: {mt5_path}"
        
        try:
            # Check if Wine can run the executable
            result = subprocess.run(['wine', mt5_path, '/?'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return True, "MT5 can be launched through Wine"
            else:
                return False, f"Wine execution failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Wine execution timed out"
        except FileNotFoundError:
            return False, "Wine not found"
        except Exception as e:
            return False, f"Wine test failed: {str(e)}"
    
    def generate_env_config(self, selected_path: str) -> str:
        """Generate .env configuration for selected MT5 path"""
        config_lines = []
        
        # Add MT5 path configuration
        if self.system == 'windows':
            config_lines.extend([
                "# MT5 Windows Configuration",
                f"MT5_PATH_WINDOWS_CUSTOM={selected_path}",
                "MT5_AUTO_DETECT_PATH=false"
            ])
        elif self.system == 'darwin':
            config_lines.extend([
                "# MT5 macOS Configuration",
                f"MT5_PATH_MACOS_CUSTOM={selected_path}",
                "MT5_AUTO_DETECT_PATH=false"
            ])
        elif self.system == 'linux':
            config_lines.extend([
                "# MT5 Linux Configuration", 
                f"MT5_PATH_LINUX_CUSTOM={selected_path}",
                "MT5_AUTO_DETECT_PATH=false"
            ])
        
        return '\n'.join(config_lines)


def main():
    """Main function for MT5 setup utility"""
    print("🔧 MT5 Setup Utility - Cross-Platform Detection")
    print("=" * 60)
    
    detector = MT5PathDetector()
    system_name = platform.system()
    
    print(f"🖥️  Operating System: {system_name}")
    print(f"🔍 Searching for MT5 installations...")
    
    # Detect all installations
    installations = detector.detect_all_mt5_installations()
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Found {len(installations['found_paths'])} MT5 installation(s)")
    print(f"   Found {len(installations['data_paths'])} data directory(ies)")
    
    if installations['found_paths']:
        print(f"\n✅ MT5 Installations Found:")
        for i, path in enumerate(installations['found_paths'], 1):
            print(f"   {i}. {path}")
            
            # Test connection
            success, message = detector.test_mt5_connection(path)
            status = "✅ Working" if success else f"❌ {message}"
            print(f"      Status: {status}")
    
    if installations['data_paths']:
        print(f"\n📁 Data Directories:")
        for path in installations['data_paths']:
            print(f"   • {path}")
    
    if installations['wine_prefixes']:
        print(f"\n🍷 Wine Prefixes:")
        for prefix in installations['wine_prefixes']:
            print(f"   • {prefix}")
    
    if installations['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in installations['recommendations']:
            print(f"   • {rec}")
    
    # Interactive selection
    if installations['found_paths']:
        print(f"\n🎯 Configuration Generation:")
        
        if len(installations['found_paths']) == 1:
            selected_path = installations['found_paths'][0]
            print(f"   Using: {selected_path}")
        else:
            print("   Select MT5 installation to use:")
            for i, path in enumerate(installations['found_paths'], 1):
                print(f"   {i}. {path}")
            
            try:
                choice = int(input("   Enter choice (1-{}): ".format(len(installations['found_paths']))))
                selected_path = installations['found_paths'][choice - 1]
            except (ValueError, IndexError):
                selected_path = installations['found_paths'][0]
                print(f"   Using default: {selected_path}")
        
        # Generate configuration
        config = detector.generate_env_config(selected_path)
        
        print(f"\n📝 Generated Configuration:")
        print("   Add these lines to your .env file:")
        print("   " + "-" * 40)
        for line in config.split('\n'):
            print(f"   {line}")
        print("   " + "-" * 40)
        
        # Save to file option
        save_config = input("\n💾 Save configuration to mt5_config.env? (y/n): ").lower().strip()
        if save_config == 'y':
            with open('mt5_config.env', 'w') as f:
                f.write(config)
            print("   ✅ Configuration saved to mt5_config.env")
            print("   📋 Copy these lines to your .env file")
    
    else:
        print(f"\n❌ No MT5 installations found!")
        print(f"📥 Installation Guide:")
        
        if system_name == "Windows":
            print("   1. Download MT5: https://www.metatrader5.com/en/download")
            print("   2. Run installer as Administrator")
            print("   3. Install to default location")
            print("   4. Run this utility again")
        
        elif system_name == "Darwin":  # macOS
            print("   1. Install Wine: brew install wine-stable")
            print("   2. Download MT5 Windows version")
            print("   3. Install: wine MetaTrader5Setup.exe")
            print("   4. Run this utility again")
        
        elif system_name == "Linux":
            print("   1. Install Wine: sudo apt install wine")
            print("   2. Download MT5 Windows version")
            print("   3. Install: wine MetaTrader5Setup.exe")
            print("   4. Run this utility again")
    
    print(f"\n🎉 MT5 Setup Utility Complete!")


if __name__ == "__main__":
    main()