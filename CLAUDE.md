# Ad-platform Incrementality Analysis

## Data Dictionary

### AUCTIONS_USERS (AUCTIONS)
- `AUCTION_ID` (Binary): Unique auction identifier
- `OPAQUE_USER_ID` (Varchar): Anonymized user identifier
- `CREATED_AT` (Timestamp_NTZ): Auction creation time

### AUCTIONS_RESULTS (BIDS)
- `AUCTION_ID` (Binary): Links to AUCTIONS_USERS
- `VENDOR_ID` (Binary): Advertiser/vendor identifier
- `CAMPAIGN_ID` (Binary): Campaign identifier
- `PRODUCT_ID` (Varchar): Product being advertised
- `RANKING` (Number): Bid rank (1=highest)
- `IS_WINNER` (Boolean): Whether bid won impression slot
- `CREATED_AT` (Timestamp_NTZ): Bid creation time

### IMPRESSIONS
- `INTERACTION_ID` (Varchar): Unique impression identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product shown
- `USER_ID` (Varchar): User who saw impression
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Impression time

### CLICKS
- `INTERACTION_ID` (Varchar): Unique click identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product clicked
- `USER_ID` (Varchar): User who clicked
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Click time

### PURCHASES
- `PURCHASE_ID` (Varchar): Unique purchase identifier
- `PURCHASED_AT` (Timestamp_NTZ): Purchase time
- `PRODUCT_ID` (Varchar): Product purchased
- `QUANTITY` (Number): Units purchased
- `UNIT_PRICE` (Number): Price per unit
- `USER_ID` (Varchar): Purchaser
- `PURCHASE_LINE` (Number): Line item number

## Guidelines

- All code must lead to consolidated .txt reports. 
- Do not generate graphs, csvs, logs, etc. for inter
- do not add a summary section. 