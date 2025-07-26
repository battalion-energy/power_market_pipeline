"""ERCOT Selenium-based data scraper for historical data."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential
from webdriver_manager.chrome import ChromeDriverManager

from .constants import DATA_PRODUCTS


class ERCOTSeleniumClient:
    """Client for scraping ERCOT data using Selenium."""
    
    def __init__(self, download_dir: str, username: Optional[str] = None, password: Optional[str] = None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.username = username
        self.password = password
        self.driver = None
    
    def setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome driver with appropriate options."""
        chrome_options = Options()
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)
    
    def login(self):
        """Log in to ERCOT if credentials are provided."""
        if not self.username or not self.password:
            return
        
        try:
            # Wait for login form
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "email"))
            )
            email_field.send_keys(self.username)
            
            password_field = self.driver.find_element(By.ID, "password")
            password_field.send_keys(self.password)
            
            login_button = self.driver.find_element(By.ID, "next")
            login_button.click()
            
            # Wait for login to complete
            WebDriverWait(self.driver, 10).until(
                EC.url_contains("ercot.com")
            )
            
            # Handle agreement dialog if present
            try:
                agree_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Agree')]"))
                )
                agree_button.click()
            except TimeoutException:
                pass  # Agreement might not appear
                
        except Exception as e:
            raise Exception(f"Login failed: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def click_download_button(self, button_element) -> bool:
        """Click a download button with retry logic."""
        try:
            # Scroll into view
            self.driver.execute_script("arguments[0].scrollIntoView(true);", button_element)
            time.sleep(0.5)
            
            # Try JavaScript click first
            self.driver.execute_script("arguments[0].click();", button_element)
            return True
        except Exception:
            # Fallback to regular click
            try:
                button_element.click()
                return True
            except Exception as e:
                print(f"Failed to click button: {str(e)}")
                return False
    
    def wait_for_download(self, filename: str, timeout: int = 60) -> bool:
        """Wait for a file download to complete."""
        file_path = self.download_dir / filename
        partial_path = self.download_dir / f"{filename}.crdownload"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if file_path.exists() and file_path.stat().st_size > 0:
                return True
            if not partial_path.exists():
                time.sleep(1)
                if file_path.exists():
                    return True
            time.sleep(1)
        
        return False
    
    def scrape_data_product(
        self, 
        product_key: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[str]:
        """Scrape a specific ERCOT data product."""
        if product_key not in DATA_PRODUCTS:
            raise ValueError(f"Unknown product key: {product_key}")
        
        product = DATA_PRODUCTS[product_key]
        downloaded_files = []
        
        try:
            self.driver = self.setup_driver()
            self.driver.get(product["url"])
            
            if product["requires_auth"]:
                self.login()
            
            # Wait for page to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//td/a[contains(text(), 'Download')]"))
            )
            
            # Set items per page to maximum
            try:
                select_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select.me-2.w-auto"))
                )
                select = Select(select_element)
                select.select_by_value("200")
                time.sleep(2)  # Wait for page to refresh
            except Exception:
                pass  # Pagination might not be available
            
            # Process downloads
            downloaded_files = self._process_downloads(start_date, end_date)
            
        finally:
            if self.driver:
                self.driver.quit()
        
        return downloaded_files
    
    def _process_downloads(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Process downloads within date range."""
        downloaded_files = []
        processed_dates = set()
        
        while True:
            rows = self.driver.find_elements(By.XPATH, "//table//tr")
            page_has_data = False
            
            for row in rows:
                try:
                    # Extract date from row
                    date_cells = row.find_elements(By.XPATH, ".//td[3]")
                    if not date_cells:
                        continue
                    
                    date_text = date_cells[0].text.strip()
                    if not date_text:
                        continue
                    
                    try:
                        file_date = datetime.strptime(date_text.split()[0], "%m/%d/%Y")
                    except:
                        continue
                    
                    # Check date range
                    if file_date.date() < start_date.date():
                        continue
                    if file_date.date() > end_date.date():
                        return downloaded_files  # We've gone past our range
                    
                    # Skip if already processed
                    if file_date.date() in processed_dates:
                        continue
                    
                    # Find download link
                    download_links = row.find_elements(By.XPATH, ".//a[contains(text(), 'Download')]")
                    if not download_links:
                        continue
                    
                    # Get filename
                    filename_cells = row.find_elements(By.XPATH, ".//td[2]")
                    if filename_cells:
                        filename = filename_cells[0].text.strip()
                    else:
                        filename = f"ercot_data_{file_date.strftime('%Y%m%d')}.csv"
                    
                    # Check if file already exists
                    if (self.download_dir / filename).exists():
                        print(f"Skipping existing file: {filename}")
                        processed_dates.add(file_date.date())
                        continue
                    
                    # Download file
                    if self.click_download_button(download_links[0]):
                        if self.wait_for_download(filename):
                            downloaded_files.append(filename)
                            processed_dates.add(file_date.date())
                            print(f"Downloaded: {filename}")
                        else:
                            print(f"Download timeout: {filename}")
                    
                    page_has_data = True
                    
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            # Check for next page
            try:
                next_button = self.driver.find_element(
                    By.XPATH, "//button[contains(@class, 'page-link') and contains(text(), 'Next')]"
                )
                if next_button.is_enabled():
                    next_button.click()
                    time.sleep(2)  # Wait for page load
                else:
                    break
            except NoSuchElementException:
                break
        
        return downloaded_files