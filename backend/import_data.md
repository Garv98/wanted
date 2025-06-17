# Importing Cleaned Data into MongoDB Atlas

1. **Install MongoDB Database Tools**  
   Download and install from:  
   https://www.mongodb.com/try/download/database-tools

2. **Get your Atlas connection string**  
   Example:  
   ```
   mongodb+srv://<user>:<password>@cluster0.mongodb.net/
   ```

3. **Import your cleaned data**  
   Open a terminal in the `data` folder and run:
   ```
   mongoimport --uri "<your_connection_string>" \
     --db crime_db --collection crimes \
     --file crime_dataset_india_cleaned.json --jsonArray
   ```

4. **Verify import in Atlas UI**  
   Go to your cluster, select the `crime_db` database and `crimes` collection to see your data.

## Troubleshooting: `'mongoimport' is not recognized as an internal or external command`

You need to install the MongoDB Database Tools:

1. Download the tools from:  
   https://www.mongodb.com/try/download/database-tools

2. Extract the downloaded archive.

3. Add the extracted folder (containing `mongoimport.exe`) to your system PATH:
   - Open "Edit the system environment variables" on Windows.
   - Click "Environment Variables".
   - Under "System variables", find and edit the `Path` variable.
   - Add the path to the folder containing `mongoimport.exe`.
   - Click OK to save.

4. Open a new terminal and run:
   ```
   mongoimport --version
   ```
   You should see the version info if installed correctly.

5. Now retry the import command:
   ```
   mongoimport --uri "<your_connection_string>" \
     --db crime_db --collection crimes \
     --file crime_dataset_india_cleaned.json --jsonArray
   ```
