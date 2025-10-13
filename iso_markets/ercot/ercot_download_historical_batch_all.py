from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
import time
import os
import glob
import zipfile
import csv
import json
import re
from datetime import datetime

def setup_chrome_driver(download_dir):
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        "safebrowsing.disable_download_protection": True,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
        "profile.managed_default_content_settings.images": 2
    })

    # Add arguments to prevent download interruption
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")

    return webdriver.Chrome(options=chrome_options)

def login(driver, username, password):
    # Wait for the email field and enter the username
    email_field = WebDriverWait(driver, 45).until(
        EC.presence_of_element_located((By.ID, "email"))
    )
    email_field.send_keys(username)

    # Find the password field and enter the password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)

    # Find and click the login button
    login_button = driver.find_element(By.ID, "next")
    login_button.click()

    # Wait for the login process to complete
    WebDriverWait(driver, 10).until(
        EC.url_contains("ercot.com")
    )

    # Wait for and click the "I Agree" button in the dialog
    try:
        agree_button = WebDriverWait(driver, 45).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Agree')]"))
        )
        agree_button.click()
    except TimeoutException:
        print("Timed out waiting for 'I Agree' button. It might not have appeared.")

def wait_for_download_buttons(driver, timeout):
    try:
        download_buttons = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
        )
        return download_buttons
    except TimeoutException:
        return None

def wait_for_download_modal_to_disappear(driver, timeout=120):
    try:
        # First, check if the modal is present
        modal = driver.find_elements(By.ID, "loading-modal")
        if not modal:
            print("Download modal not found. Continuing...")
            return True
        else:
            print("Download modal found. Waiting for it to disappear...")

        # If the modal is present, wait for it to disappear
        WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.ID, "loading-modal"))
        )
        print("Download modal disappeared.")
        return True
    except TimeoutException:
        print("Download modal did not disappear within the expected time. Continuing...")
        return False

def click_button_safely(driver, button, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            # Scroll the button into view
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            # Try to click the button
            button.click()
            return True
        except ElementClickInterceptedException:
            if attempt == max_attempts - 1:
                print(f"Failed to click button after {max_attempts} attempts.")
                return False
            print(f"Click intercepted, retrying... (Attempt {attempt + 1})")
            time.sleep(0.5)  # Minimal wait before retry
        except Exception as e:
            print(f"An error occurred while trying to click the button: {str(e)}")
            return False
    return False

def file_exists_and_not_empty(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def verify_zip_integrity(download_dir):
    """Check the integrity and contents of downloaded ZIP files"""
    zip_files = glob.glob(os.path.join(download_dir, '*.zip'))

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test the ZIP file integrity
                zip_ref.testzip()

                # List contents
                file_list = zip_ref.namelist()
                total_size = sum(zip_ref.getinfo(name).file_size for name in file_list)

                print(f"ZIP: {os.path.basename(zip_path)}")
                print(f"  Files: {len(file_list)}")
                print(f"  Total uncompressed size: {total_size:,} bytes")
                print(f"  ZIP file size: {os.path.getsize(zip_path):,} bytes")

                # Check if ZIP seems incomplete (too small or only 1 file when expecting more)
                if len(file_list) == 1 and total_size < 1000:  # Less than 1KB uncompressed
                    print(f"  WARNING: {os.path.basename(zip_path)} may be incomplete!")

        except zipfile.BadZipFile:
            print(f"ERROR: {os.path.basename(zip_path)} is corrupted!")
        except Exception as e:
            print(f"ERROR checking {os.path.basename(zip_path)}: {e}")

def check_downloads_complete(download_dir, timeout=600):
    # Check for any .crdownload or .tmp files
    incomplete_downloads = glob.glob(os.path.join(download_dir, '*.crdownload')) + glob.glob(os.path.join(download_dir, '*.tmp'))

    # Wait up to timeout seconds for downloads to complete (default 10 minutes)
    start_time = time.time()
    while incomplete_downloads and (time.time() - start_time) < timeout:
        print(f"Waiting for downloads to complete... {len(incomplete_downloads)} files still downloading")
        time.sleep(1)  # Check every second for download completion
        incomplete_downloads = glob.glob(os.path.join(download_dir, '*.crdownload')) + glob.glob(os.path.join(download_dir, '*.tmp'))

    return len(incomplete_downloads) == 0

def check_for_rate_limit_or_error(driver):
    """
    Check if the page shows rate limiting or server error messages.
    Returns tuple: (is_rate_limited, error_message)
    """
    try:
        # Check page source for common rate limit / error indicators
        page_source = driver.page_source.lower()

        rate_limit_indicators = [
            "too many requests",
            "rate limit",
            "slow down",
            "429",
            "service unavailable",
            "503",
            "server error",
            "500",
            "please try again later",
            "temporarily unavailable"
        ]

        for indicator in rate_limit_indicators:
            if indicator in page_source:
                print(f"Detected rate limit/error indicator: '{indicator}'")
                return True, indicator

        # Check for error alert boxes
        try:
            alerts = driver.find_elements(By.CSS_SELECTOR, ".alert-danger, .alert-warning, .error-message")
            if alerts:
                for alert in alerts:
                    text = alert.text.lower()
                    for indicator in rate_limit_indicators:
                        if indicator in text:
                            print(f"Found error alert: {alert.text[:100]}")
                            return True, alert.text[:100]
        except:
            pass

        return False, None

    except Exception as e:
        print(f"Error checking for rate limit: {str(e)}")
        return False, None

def detect_max_page_size(driver):
    """
    Detect the maximum available page size for the current data product.
    Tries page sizes in descending order: 500, 200, 100, 50, 25, 10
    Returns the largest available page size.
    """
    possible_sizes = [500, 200, 100, 50, 25, 10]

    try:
        select_element = WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select.me-2.w-auto"))
        )
        select = Select(select_element)

        # Get all available options
        available_options = [int(option.get_attribute('value')) for option in select.options]
        print(f"Available page sizes: {available_options}")

        # Find the largest available size
        for size in possible_sizes:
            if size in available_options:
                print(f"Detected max page size: {size}")
                return size

        # If none of the expected sizes are available, return the largest available
        if available_options:
            max_size = max(available_options)
            print(f"Using largest available page size: {max_size}")
            return max_size

        print("No page size options found, defaulting to 10")
        return 10

    except Exception as e:
        print(f"Failed to detect page size: {str(e)}, defaulting to 100")
        return 100

def convert_url_to_archive(info_url):
    """
    Convert ERCOT data product info URL to archive URL.
    Example: https://www.ercot.com/mp/data-products/data-product-details?id=np6-905-cd
    Becomes: https://data.ercot.com/data-product-archive/NP6-905-CD
    """
    match = re.search(r'id=([^&]+)', info_url)
    if match:
        product_id = match.group(1).upper()
        archive_url = f"https://data.ercot.com/data-product-archive/{product_id}"
        return archive_url
    return None

def sanitize_directory_name(name):
    """Sanitize the dataset name to create a valid directory name"""
    # Remove or replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace multiple spaces with single underscore
    name = re.sub(r'\s+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def load_progress(progress_file):
    """Load progress from JSON file"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}

def save_progress(progress_file, progress_data):
    """Save progress to JSON file"""
    progress_data["last_updated"] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def scrape_and_download_all_pages(url, download_dir, pageSize=None, driver=None, close_driver=True):
    """
    Scrape and download all data by paginating through all available pages.
    Starts from page 1 (most recent data) and continues until no more pages.

    Args:
        url: The ERCOT data URL
        download_dir: Directory to save downloads
        pageSize: Page size to use (if None, will auto-detect)
        driver: Existing WebDriver instance (if None, will create new one)
        close_driver: Whether to close the driver when done (default: True)
    """
    print(f"\n{'='*80}")
    print(f"Fetching all historical data")
    print(f"Download directory: {download_dir}")
    print(f"URL: {url}")
    print(f"{'='*80}\n")

    os.makedirs(download_dir, exist_ok=True)

    # Create or reuse driver
    driver_created = False
    if driver is None:
        driver = setup_chrome_driver(download_dir)
        driver_created = True

    try:
        driver.get(url)

        # Only login if we created a new driver
        if driver_created:
            login(driver, "enrico.ladendorf@battalion.energy", "52eDnRryUrA5fN6")

        # Wait for the page to load
        try:
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
            )
        except TimeoutException:
            print("ERROR: Timed out waiting for download buttons. This product may not have downloadable data.")
            return False

        # Auto-detect or set page size
        if pageSize is None:
            pageSize = detect_max_page_size(driver)

        # Select the detected page size
        try:
            select_element = WebDriverWait(driver, 45).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select.me-2.w-auto"))
            )
            select = Select(select_element)
            select.select_by_value(str(pageSize))
            print(f"Selected page size: {pageSize}")
        except Exception as e:
            print(f"Failed to select page size {pageSize}: {str(e)}")

        page_count = 0
        max_pages = 10000  # Safety limit to prevent infinite loops
        backoff_time = 5  # Initial backoff time in seconds
        max_backoff = 300  # Maximum backoff time (5 minutes)
        consecutive_failures = 0
        max_consecutive_failures = 5

        while page_count < max_pages:
            # Check for rate limiting before proceeding
            is_rate_limited, error_msg = check_for_rate_limit_or_error(driver)
            if is_rate_limited:
                print(f"\n!!! RATE LIMIT DETECTED: {error_msg} !!!")
                print(f"Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)  # Only sleep for rate limiting
                backoff_time = min(backoff_time * 2, max_backoff)  # Exponential backoff
                driver.refresh()
                continue

            try:
                # Wait for the All checkbox to be clickable
                try:
                    checkbox = WebDriverWait(driver, 45).until(
                        EC.element_to_be_clickable((By.ID, "checkAllBox"))
                    )

                    # Check if the checkbox is already selected
                    if not checkbox.is_selected():
                        checkbox.click()
                        print("Selected All checkbox")
                except Exception as e:
                    print(f"Failed to select All checkbox: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Too many consecutive failures ({consecutive_failures}). Giving up on this dataset.")
                        break
                    # Wait and retry
                    print(f"Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, max_backoff)
                    continue

                # Click the Download button
                try:
                    download_button = WebDriverWait(driver, 45).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
                    )
                    click_button_safely(driver, download_button)
                    print(f"Page {page_count + 1}: Download button clicked successfully.")
                except Exception as e:
                    print(f"Failed to click download button: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Too many consecutive failures ({consecutive_failures}). Giving up on this dataset.")
                        break
                    print(f"Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, max_backoff)
                    continue

                # Wait for the download modal to disappear
                modal_success = wait_for_download_modal_to_disappear(driver, timeout=300)
                if not modal_success:
                    print("Download modal timeout. Checking for rate limiting...")
                    is_rate_limited, error_msg = check_for_rate_limit_or_error(driver)
                    if is_rate_limited:
                        print(f"Rate limit detected: {error_msg}. Backing off...")
                        time.sleep(backoff_time)  # Only sleep for rate limiting
                        backoff_time = min(backoff_time * 2, max_backoff)
                        driver.refresh()
                        continue

                # Check if downloads are complete
                if check_downloads_complete(download_dir):
                    print(f"Page {page_count + 1}: Downloads completed successfully.")
                    consecutive_failures = 0  # Reset failure counter on success
                    backoff_time = 5  # Reset backoff time on success
                else:
                    print(f"Page {page_count + 1}: Downloads may not have completed. Please check the download directory.")

                # Try to go to the next page
                try:
                    next_button = WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable((By.XPATH, "//span[@class='page-link' and text()='»']"))
                    )
                    if click_button_safely(driver, next_button):
                        page_count += 1
                        print(f"Successfully moved to page {page_count + 1}")
                    else:
                        print("Failed to click next page button. Likely reached the last page.")
                        break
                except (NoSuchElementException, TimeoutException):
                    print("No more pages to scrape or timed out waiting for next page.")
                    break

            except Exception as e:
                print(f"Unexpected error on page {page_count + 1}: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Too many consecutive failures ({consecutive_failures}). Giving up on this dataset.")
                    break
                print(f"Waiting {backoff_time} seconds before retry...")
                time.sleep(backoff_time)  # Only sleep for error recovery
                backoff_time = min(backoff_time * 2, max_backoff)

        print(f"\nCompleted scraping {page_count + 1} pages for {url}")
        return True

    except Exception as e:
        print(f"ERROR during scraping: {str(e)}")
        return False

    finally:
        if close_driver and driver_created:
            driver.quit()

def read_data_products_csv(csv_file):
    """Read the data products CSV and return a list of datasets to download"""
    datasets = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            info_url = row['URL']
            name = row['Name']
            is_secure = row['Is Secure'] == 'True'

            # Convert to archive URL
            archive_url = convert_url_to_archive(info_url)
            if archive_url:
                datasets.append({
                    'url': archive_url,
                    'name': name,
                    'is_secure': is_secure,
                    'info_url': info_url
                })

    return datasets

def main():
    # Configuration
    base_download_dir = "/pool/ssd8tb/data/iso/ERCOT_data_clean"
    csv_file = "/home/enrico/projects/power_market_pipeline/iso_markets/ercot/ercot_data_products.csv"
    progress_file = os.path.join(base_download_dir, "download_progress.json")

    # Create base directory
    os.makedirs(base_download_dir, exist_ok=True)

    # Load progress
    progress = load_progress(progress_file)

    # Read all datasets from CSV
    print("Reading data products from CSV...")
    datasets = read_data_products_csv(csv_file)
    print(f"Found {len(datasets)} data products to download\n")

    # Filter out already completed datasets
    remaining_datasets = [d for d in datasets if d['url'] not in progress['completed']]
    print(f"Already completed: {len(progress['completed'])}")
    print(f"Failed previously: {len(progress['failed'])}")
    print(f"Remaining to download: {len(remaining_datasets)}\n")

    # Download each dataset
    for i, dataset in enumerate(remaining_datasets, 1):
        print(f"\n{'#'*80}")
        print(f"# Processing dataset {i}/{len(remaining_datasets)}")
        print(f"# Name: {dataset['name']}")
        print(f"# URL: {dataset['url']}")
        print(f"{'#'*80}\n")

        # Create sanitized directory name
        dir_name = sanitize_directory_name(dataset['name'])
        download_dir = os.path.join(base_download_dir, dir_name)

        # Skip secure datasets for now (they may require special handling)
        if dataset['is_secure']:
            print(f"SKIPPING: This is a secure dataset (requires authentication)")
            continue

        try:
            # Download the dataset
            success = scrape_and_download_all_pages(
                url=dataset['url'],
                download_dir=download_dir,
                pageSize=None,  # Auto-detect
                driver=None,  # Create new driver for each dataset
                close_driver=True
            )

            if success:
                progress['completed'].append(dataset['url'])
                print(f"\nSUCCESS: Completed download for {dataset['name']}")
            else:
                progress['failed'].append({
                    'url': dataset['url'],
                    'name': dataset['name'],
                    'error': 'Download returned False'
                })
                print(f"\nFAILED: Could not download {dataset['name']}")

        except Exception as e:
            progress['failed'].append({
                'url': dataset['url'],
                'name': dataset['name'],
                'error': str(e)
            })
            print(f"\nERROR: Exception occurred while downloading {dataset['name']}: {str(e)}")

        finally:
            # Save progress after each dataset
            save_progress(progress_file, progress)
            print(f"\nProgress saved. Completed: {len(progress['completed'])}, Failed: {len(progress['failed'])}")

    print("\n" + "="*80)
    print("ALL DOWNLOADS COMPLETE!")
    print(f"Successfully downloaded: {len(progress['completed'])} datasets")
    print(f"Failed: {len(progress['failed'])} datasets")
    print(f"Progress file: {progress_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
