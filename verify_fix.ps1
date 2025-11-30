$payload = @{
    body = @{
        content = "test message"
        message_type = "incoming"
        sender = @{
            phone_number = "123456789"
        }
        conversation = @{
            messages = @(
                @{
                    conversation_id = 111
                }
            )
        }
    }
}

$url = "http://localhost:3000/fb7a77d6-fe52-4622-b0af-07429a5e6e45"
# No query param in URL

try {
    $response = Invoke-RestMethod -Uri $url -Method Post -Body ($payload | ConvertTo-Json -Depth 10) -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_"
    $_.Exception.Response
}
