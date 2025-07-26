"""ERCOT Selenium scraper for historical data (before Dec 11, 2023)."""

import asyncio
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv

from .constants import WEBSERVICE_CUTOFF_DATE

load_dotenv()


class ERCOTSeleniumScraper:
    """Scraper for ERCOT's historical data portal using Selenium."""
    
    BASE_URL = "https://www.ercot.com"
    LOGIN_URL = "https://www.ercot.com/account/login"
    MIS_URL = "https://www.ercot.com/mp/data-products/data-product-details?id=NP4-183-CD"
    
    def __init__(self):
        self.username = os.getenv("ERCOT_SELENIUM_USERNAME")
        self.password = os.getenv("ERCOT_SELENIUM_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError("ERCOT Selenium credentials not found in environment")
        
        self.download_dir = tempfile.mkdtemp()
        self.driver = None
        
    def _setup_driver(self):
        """Set up Chrome driver with download preferences."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Set download preferences
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def _login(self):
        """Log in to ERCOT MIS portal."""
        print("Logging in to ERCOT...")
        self.driver.get(self.LOGIN_URL)
        
        # Wait for login form
        wait = WebDriverWait(self.driver, 20)
        
        # Enter credentials
        username_field = wait.until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        username_field.send_keys(self.username)
        
        password_field = self.driver.find_element(By.ID, "password")
        password_field.send_keys(self.password)
        
        # Submit form
        login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        
        # Wait for redirect after login
        time.sleep(5)
        print("Login successful")
        
    def _navigate_to_dam_prices(self):
        """Navigate to DAM prices data product."""
        print("Navigating to DAM prices data...")
        self.driver.get(self.MIS_URL)
        
        # Wait for page to load
        wait = WebDriverWait(self.driver, 20)
        wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "data-product-details"))
        )
        
    def _download_file_for_date(self, date: datetime) -> Optional[Path]:
        """Download DAM price file for a specific date."""
        date_str = date.strftime("%Y%m%d")
        year = date.strftime("%Y")
        month = date.strftime("%m")
        
        print(f"Downloading DAM prices for {date_str}...")
        
        # Look for download link
        wait = WebDriverWait(self.driver, 10)
        
        try:
            # Find link matching the date pattern
            link_xpath = f"//a[contains(@href, 'dam_spp_{date_str}')]"
            download_link = wait.until(
                EC.presence_of_element_located((By.XPATH, link_xpath))
            )
            
            # Click download
            download_link.click()
            
            # Wait for download to complete
            expected_filename = f"dam_spp_{date_str}.zip"
            download_path = Path(self.download_dir) / expected_filename
            
            max_wait = 60
            while max_wait > 0 and not download_path.exists():
                time.sleep(1)
                max_wait -= 1
                
            if download_path.exists():
                print(f"Downloaded: {expected_filename}")
                return download_path
            else:
                print(f"Download failed for {date_str}")
                return None
                
        except Exception as e:
            print(f"Could not find download link for {date_str}: {e}")
            return None
            
    def _extract_and_parse_zip(self, zip_path: Path) -> pd.DataFrame:
        """Extract and parse DAM price data from zip file."""
        data_frames = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.endswith('.csv'):
                    with zip_file.open(file_name) as csv_file:
                        df = pd.read_csv(csv_file)
                        data_frames.append(df)
                        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
            
    async def get_dam_spp_prices(
        self,
        start_date: datetime,
        end_date: datetime,
        settlement_points: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get historical DAM SPP prices using Selenium."""
        if end_date >= WEBSERVICE_CUTOFF_DATE:
            print(f"Use WebService API for dates after {WEBSERVICE_CUTOFF_DATE}")
            end_date = WEBSERVICE_CUTOFF_DATE - timedelta(days=1)
            
        all_data = []
        
        try:
            # Set up driver
            self._setup_driver()
            
            # Login
            self._login()
            
            # Navigate to DAM prices
            self._navigate_to_dam_prices()
            
            # Download files for each date
            current_date = start_date
            while current_date <= end_date:
                # Download file
                zip_path = self._download_file_for_date(current_date)
                
                if zip_path:
                    # Extract and parse
                    df = self._extract_and_parse_zip(zip_path)
                    
                    if not df.empty:
                        # Filter by settlement points if specified
                        if settlement_points:
                            df = df[df['SettlementPoint'].isin(settlement_points)]
                            
                        all_data.append(df)
                        
                    # Clean up zip file
                    zip_path.unlink()
                    
                current_date += timedelta(days=1)
                
                # Add delay to avoid rate limiting
                time.sleep(2)
                
        finally:
            # Clean up
            if self.driver:
                self.driver.quit()
                
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            
            # Standardize column names
            result_df.rename(columns={
                'DeliveryDate': 'delivery_date',
                'HourEnding': 'hour_ending',
                'SettlementPoint': 'settlement_point',
                'SettlementPointPrice': 'spp',
                'DSTFlag': 'dst_flag'
            }, inplace=True)
            
            return result_df
        else:
            return pd.DataFrame()
            
    async def test_connection(self) -> bool:
        """Test the Selenium connection."""
        try:
            self._setup_driver()
            self._login()
            return True
        except Exception as e:
            print(f"Selenium test failed: {e}")
            return False
        finally:
            if self.driver:
                self.driver.quit()