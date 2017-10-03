# Installing Ubuntu on Razer Blade

## Preparations

### Shrinking Space of Disk

Usually, Windows 10 installer will fully uses space of disk that was remained by recovery and UEFI partitions. Run the below command to launch the **Windows Disk Manager** , you can use the search bar in *startup menu* to find and run it:

```shell
diskmgmt.msc
```

Once you open the Windows Disk Manager, right click the System Partition and select the 'shrink partition', re-plan the layout of partitions and leave enough free space for Linux.