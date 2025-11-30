$payload = @(
    @{
        body = @{
            content = "prueba de query desde URL"
            content_type = "text"
            message_type = "incoming"
            sender = @{
                name = "Rayyan"
                phone_number = "+51924172652"
                custom_attributes = @{
                    waha_whatsapp_jid = "51924172652@c.us"
                }
            }
            conversation = @{
                messages = @(
                    @{
                        conversation_id = 12345
                        attachments = $null
                        sender = @{
                            custom_attributes = @{
                                eliteseller_trigger = $false
                            }
                        }
                    }
                )
                timestamp = 1764517922
            }
            id = 76057
            account = @{
                id = 45
            }
        }
        # Query eliminado del body para probar que lo tome de la URL
    }
)

# Agregamos ?q=hardcoded_value a la URL
$response = Invoke-RestMethod -Uri "https://bot.eliteseller.app/webhook/41728f37-7ad6-4ca3-bba1-b046e05a112a?q=hola" -Method Post -Body ($payload | ConvertTo-Json -Depth 10) -ContentType "application/json"
$response | ConvertTo-Json -Depth 10
