"""
Script to scrape the ERCOT data catalog and extract all dataset URLs and names.
Saves the results to a CSV file.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
import os

def setup_chrome_driver():
    """Setup Chrome driver without download directory (we're just scraping)"""
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    return webdriver.Chrome(options=chrome_options)

def login(driver, username, password):
    """Login to ERCOT website"""
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

def scrape_ercot_catalog(url, output_csv):
    """
    Scrape the ERCOT data catalog page and extract dataset information.

    Args:
        url: The ERCOT data catalog URL
        output_csv: Path to save the CSV file with results
    """
    print(f"Scraping ERCOT catalog: {url}")

    driver = setup_chrome_driver()
    driver.get(url)

    # Login
    login(driver, "enrico.ladendorf@battalion.energy", "52eDnRryUrA5fN6")

    # Wait for the page to load
    print("Waiting for page to load...")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(5)

    # Print page title and URL for debugging
    print(f"Current page title: {driver.title}")
    print(f"Current URL: {driver.current_url}")

    # Select 200 items per page
    try:
        # Try to find the items-per-page selector
        select_element = WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select"))
        )
        select = Select(select_element)

        # Check available options
        options = [option.get_attribute('value') for option in select.options]
        print(f"Available page size options: {options}")

        select.select_by_value("200")
        print("Selected 200 items per page")
        time.sleep(3)  # Wait for page to reload
    except Exception as e:
        print(f"Failed to select 200 items per page: {str(e)}")
        print("Continuing with default page size...")

    datasets = []
    page_count = 0

    while True:
        print(f"\nProcessing page {page_count + 1}...")

        # Wait for the table/list to load
        time.sleep(3)

        # Try to find all links on the page that go to data-product-archive
        try:
            # Find all links on the page
            all_links = driver.find_elements(By.TAG_NAME, "a")
            print(f"Found {len(all_links)} total links on page")

            archive_links_on_page = 0

            for link in all_links:
                try:
                    href = link.get_attribute("href")

                    if href and "data-product-archive" in href.lower():
                        dataset_name = link.text.strip()

                        # Only process if we have a non-empty name
                        if dataset_name:
                            # Store dataset info
                            dataset_info = {
                                "url": href,
                                "name": dataset_name
                            }

                            # Avoid duplicates
                            if not any(d["url"] == href for d in datasets):
                                datasets.append(dataset_info)
                                print(f"  Found: {dataset_name} -> {href}")
                                archive_links_on_page += 1

                except Exception as e:
                    # Skip problematic links
                    continue

            print(f"Found {archive_links_on_page} data-product-archive links on this page")

        except Exception as e:
            print(f"Error processing page: {str(e)}")

        # Try to go to next page
        try:
            next_button = driver.find_element(By.XPATH, "//a[@class='page-link' and contains(text(), 'Next')]")
            if "disabled" in next_button.get_attribute("class"):
                print("Reached last page")
                break

            next_button.click()
            page_count += 1
            time.sleep(3)
        except (NoSuchElementException, TimeoutException):
            # Try alternative next button selector
            try:
                next_button = driver.find_element(By.XPATH, "//span[@class='page-link' and text()='»']")
                parent = next_button.find_element(By.XPATH, "..")

                if "disabled" in parent.get_attribute("class"):
                    print("Reached last page")
                    break

                next_button.click()
                page_count += 1
                time.sleep(3)
            except:
                print("No more pages found")
                break

    driver.quit()

    # Save to CSV
    print(f"\nSaving {len(datasets)} datasets to {output_csv}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        if datasets:
            fieldnames = ['url', 'name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for dataset in datasets:
                writer.writerow(dataset)

    print(f"Done! Scraped {len(datasets)} datasets across {page_count + 1} pages")
    return datasets

if __name__ == "__main__":
    # Main catalog page URL
    catalog_url = "https://data.ercot.com/"

    # Output CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, "ercot_datasets_catalog.csv")

    # Scrape the catalog
    datasets = scrape_ercot_catalog(catalog_url, output_csv)

    print("\n" + "="*50)
    print(f"Successfully scraped {len(datasets)} datasets")
    print(f"Results saved to: {output_csv}")
