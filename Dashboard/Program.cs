using Dashboard.Components;
using Dashboard.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddHttpClient(); // provide HttpClient for AlertService
builder.Services.AddSingleton<AlertService>();

var app = builder.Build();

// Start WebSocket listener for live alerts
var alertService = app.Services.GetRequiredService<AlertService>();
_ = Task.Run(() => alertService.StartWebSocketAsync());

// Configure the HTTP request pipeline
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

// app.UseHttpsRedirection();
app.UseStaticFiles(); // serve wwwroot assets

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
