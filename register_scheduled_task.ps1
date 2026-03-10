# E-Netra V4 — Register Windows Scheduled Task
# Run this ONCE as Administrator to schedule daily pipeline + dashboard rebuild
#
# Schedule: every day at 06:00 local time
# The task rebuilds the dashboard by running watch_rebuild.py --no-serve (just rebuild, no server)

$taskName = "ENetra-DashboardRebuild"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python).Source
$script = Join-Path $scriptRoot "scripts\watch_rebuild.py"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "-X utf8 `"$script`" --no-serve --interval 99999" `
    -WorkingDirectory $scriptRoot

# Trigger: daily at 06:00
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -WakeToRun

Register-ScheduledTask `
    -TaskName $taskName `
    -Action   $action `
    -Trigger  $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Host "✓ Task '$taskName' registered — runs daily at 06:00" -ForegroundColor Green
Write-Host "  To trigger manually: Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor Gray
