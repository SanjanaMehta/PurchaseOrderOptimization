
SELECT * FROM po_lifecycle_vendor
left join surgical_inventory
on surgical_inventory.Item_ID = po_lifecycle_vendor.Item_ID;